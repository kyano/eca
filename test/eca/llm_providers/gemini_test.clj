(ns eca.llm-providers.gemini-test
  (:require
   [cheshire.core :as json]
   [clojure.string :as string]
   [clojure.test :refer [deftest is testing]]
   [eca.client-test-helpers :refer [with-client-proxied]]
   [eca.llm-providers.gemini :as llm-providers.gemini]
   [hato.client :as http]
   [matcher-combinators.matchers :as m]
   [matcher-combinators.test :refer [match?]]))

(defn ^:private sse-stream
  "Builds an InputStream of unnamed `data: {...}` SSE chunks, mirroring what
   Gemini's streamGenerateContent endpoint sends (no `event:` line)."
  [& chunks]
  (java.io.ByteArrayInputStream.
   (.getBytes ^String (apply str (map #(str "data: " (json/generate-string %) "\n\n") chunks))
              "UTF-8")))

(defn ^:private collecting-callbacks [events*]
  {:on-message-received #(swap! events* conj [:msg %])
   :on-error #(swap! events* conj [:error %])
   :on-reason #(swap! events* conj [:reason %])
   :on-prepare-tool-call #(swap! events* conj [:prepare %])
   :on-tools-called (constantly nil)
   :on-usage-updated #(swap! events* conj [:usage %])
   :on-server-web-search #(swap! events* conj [:web-search %])})

(deftest normalize-messages-test
  (testing "user message formats to user role with text parts"
    (is (match? [{:role "user"
                  :parts [{:text "Hello world"}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "user" :content "Hello world"}]))))

  (testing "assistant message formats to model role with text parts"
    (is (match? [{:role "model"
                  :parts [{:text "Hello from assistant"}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "assistant" :content "Hello from assistant"}]))))

  (testing "system messages are dropped from contents"
    (is (= []
           (#'llm-providers.gemini/normalize-messages
            [{:role "system" :content "You are helpful"}]))))

  (testing "foreign API messages are dropped"
    (is (= []
           (#'llm-providers.gemini/normalize-messages
            [{:role "tool_call"
              :content {:api :anthropic
                        :id "call_1"
                        :full-name "search"
                        :arguments {:q "test"}}}
             {:role "tool_call"
              :content {:api :openai-chat
                        :id "call_2"
                        :full-name "search"
                        :arguments {:q "test"}}}
             {:role "reason"
              :content {:api :anthropic
                        :text "Thinking..."}}
             {:role "tool_call_output"
              :content {:api :anthropic
                        :id "call_1"
                        :full-name "search"
                        :output {:contents [{:type :text :text "done"}]}}}]))))

  (testing "tool_call message with thought-signature formats to model role with functionCall and thoughtSignature"
    (is (match? [{:role "model"
                  :parts [{:functionCall {:name "search_files"
                                          :args {:query "hello"}}
                           :thoughtSignature "sig-123"}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "tool_call"
                   :content {:id "call_1"
                             :full-name "search_files"
                             :arguments {:query "hello"}
                             :thought-signature "sig-123"}}]))))

  (testing "tool_call_output message formats to user role with functionResponse"
    (is (match? [{:role "user"
                  :parts [{:functionResponse {:name "search_files"
                                              :response {:result "file1.txt\n"}}}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "tool_call_output"
                   :content {:id "call_1"
                             :full-name "search_files"
                             :output {:contents [{:type :text :text "file1.txt"}]}}}]))))

  (testing "reason message with external-id formats to model role with thought true and thoughtSignature"
    (is (match? [{:role "model"
                  :parts [{:text "Let me think"
                           :thought true
                           :thoughtSignature "sig-123"}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "reason"
                   :content {:text "Let me think"
                             :external-id "sig-123"}}]))))

  (testing "image parts when supports-image? is true"
    (is (match? [{:role "user"
                  :parts [{:text "describe this image"}
                          {:inlineData {:mimeType "image/png"
                                        :data "base64data=="}}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "user"
                   :content [{:type "text" :text "describe this image"}
                             {:type "image" :media-type "image/png" :base64 "base64data=="}]}]
                 true))))

  (testing "image parts dropped when supports-image? is false"
    (is (match? [{:role "user"
                  :parts [{:text "describe this image"}]}]
                (#'llm-providers.gemini/normalize-messages
                 [{:role "user"
                   :content [{:type "text" :text "describe this image"}
                             {:type "image" :media-type "image/png" :base64 "base64data=="}]}]
                 false)))))

(deftest extract-system-instruction-test
  (testing "extracts system role messages and instructions into {:parts [{:text \"...\"}]}"
    (is (= {:parts [{:text "System message\nInstruction prompt"}]}
           (#'llm-providers.gemini/extract-system-instruction
            [{:role "system" :content "System message"}]
            "Instruction prompt")))
    (is (= {:parts [{:text "Instruction only"}]}
           (#'llm-providers.gemini/extract-system-instruction
            []
            "Instruction only")))
    (is (= {:parts [{:text "System message only"}]}
           (#'llm-providers.gemini/extract-system-instruction
            [{:role "system" :content "System message only"}]
            ""))))

  (testing "returns nil when no system messages and instructions are blank"
    (is (nil? (#'llm-providers.gemini/extract-system-instruction [] "")))
    (is (nil? (#'llm-providers.gemini/extract-system-instruction [] nil)))
    (is (nil? (#'llm-providers.gemini/extract-system-instruction [] "   ")))
    (is (nil? (#'llm-providers.gemini/extract-system-instruction
               [{:role "user" :content "Hello"}]
               "")))))

(deftest sanitize-schema-for-gemini-test
  (testing "strips unsupported keys (:additionalProperties, :$schema, :exclusiveMinimum, :exclusiveMaximum)"
    (let [schema {:type "object"
                  :$schema "http://json-schema.org/draft-07/schema#"
                  :additionalProperties false
                  :properties {:count {:type "integer"
                                       :exclusiveMinimum 0
                                       :exclusiveMaximum 100}}}]
      (is (= {:type "object"
              :properties {:count {:type "integer"}}}
             (#'llm-providers.gemini/sanitize-schema-for-gemini schema)))))

  (testing "normalizes nullable union types: {:type [\"integer\" \"null\"] :description \"Task ID\"} -> {:type \"integer\" :nullable true :description \"Task ID\"}"
    (let [schema {:type ["integer" "null"]
                  :description "Task ID"}]
      (is (= {:type "integer"
              :nullable true
              :description "Task ID"}
             (#'llm-providers.gemini/sanitize-schema-for-gemini schema)))))

  (testing "drops :type if [:type [\"null\"]]"
    (let [schema {:type ["null"]
                  :description "Just null"}]
      (is (= {:description "Just null"}
             (#'llm-providers.gemini/sanitize-schema-for-gemini schema)))))

  (testing "keeps first type if [:type [\"integer\" \"string\"]]"
    (let [schema {:type ["integer" "string"]
                  :description "Union type"}]
      (is (= {:type "integer"
              :description "Union type"}
             (#'llm-providers.gemini/sanitize-schema-for-gemini schema)))))

  (testing "keeps first type and adds :nullable true if [:type [\"integer\" \"string\" \"null\"]]"
    (let [schema {:type ["integer" "string" "null"]
                  :description "Nullable union"}]
      (is (= {:type "integer"
              :nullable true
              :description "Nullable union"}
             (#'llm-providers.gemini/sanitize-schema-for-gemini schema)))))

  (testing "deep-nesting stack overflow protection: returns fallback {:type \"object\" :properties {}} for deeply nested schemas (depth 50,000)"
    (let [deep-schema (reduce (fn [acc _]
                                {:type "object"
                                 :properties {:nested acc}})
                              {:type "string"}
                              (range 50000))
          result (#'llm-providers.gemini/sanitize-schema-for-gemini deep-schema)]
      (is (map? result))
      (is (= "object" (:type result)))
      (is (match? {:type "object" :properties (m/pred map?)} result)))))

(deftest build-body-test
  (testing "builds contents, systemInstruction, tools (as [{:functionDeclarations [...]}]), generationConfig"
    (let [body (#'llm-providers.gemini/prepare-body
                {:messages [{:role "user" :content "hello"}]
                 :instructions "System prompt"
                 :tools [{:full-name "search"
                          :description "Search files"
                          :parameters {:type "object"
                                       :properties {:query {:type ["string" "null"]}}
                                       :additionalProperties false}}]
                 :max-output-tokens 4096})]
      (is (match? {:contents [{:role "user" :parts [{:text "hello"}]}]
                   :systemInstruction {:parts [{:text "System prompt"}]}
                   :tools [{:functionDeclarations [{:name "search"
                                                   :description "Search files"
                                                   :parameters {:type "object"
                                                                :properties {:query {:type "string"
                                                                                     :nullable true}}}}]}]
                   :generationConfig {:maxOutputTokens 4096}}
                  body))))

  (testing "when reason? is true, sets :generationConfig {:maxOutputTokens 32000 :thinkingConfig {:includeThoughts true}}"
    (let [body (#'llm-providers.gemini/prepare-body
                {:messages [{:role "user" :content "think about this"}]
                 :reason? true})]
      (is (= {:maxOutputTokens 32000
              :thinkingConfig {:includeThoughts true}}
             (:generationConfig body)))))

  (testing "deep-merges with extraPayload and extra-payload"
    (let [body (#'llm-providers.gemini/prepare-body
                {:messages [{:role "user" :content "think harder"}]
                 :reason? true
                 :extra-payload {:generationConfig {:thinkingConfig {:thinkingLevel "HIGH"}}
                                 :safetySettings [{:category "HARM_CATEGORY_HATE_SPEECH"
                                                   :threshold "BLOCK_NONE"}]}})]
      (is (match? {:generationConfig {:maxOutputTokens 32000
                                      :thinkingConfig {:includeThoughts true
                                                       :thinkingLevel "HIGH"}}
                   :safetySettings [{:category "HARM_CATEGORY_HATE_SPEECH"
                                     :threshold "BLOCK_NONE"}]}
                  body)))))

(deftest prepare-body-web-search-test
  (testing "when :web-search true is passed, :tools contains [{:googleSearch {}} {:urlContext {}}]"
    (let [body (#'llm-providers.gemini/prepare-body
                {:messages [{:role "user" :content "what is the weather today?"}]
                 :web-search true})]
      (is (= [{:googleSearch {}} {:urlContext {}}]
             (:tools body)))))

  (testing "when both :tools and :web-search true are provided, :tools contains functionDeclarations, googleSearch, and urlContext"
    (let [body (#'llm-providers.gemini/prepare-body
                {:messages [{:role "user" :content "search files and web"}]
                 :tools [{:full-name "search_files"
                          :description "Search files"
                          :parameters {:type "object"}}]
                 :web-search true})]
      (is (= [{:functionDeclarations [{:name "search_files"
                                       :description "Search files"
                                       :parameters {:type "object"}}]}
              {:googleSearch {}}
              {:urlContext {}}]
             (:tools body)))))

  (testing "when :web-search is false or nil, :tools does not include :googleSearch or :urlContext"
    (let [body-nil (#'llm-providers.gemini/prepare-body
                    {:messages [{:role "user" :content "hello"}]
                     :web-search nil})
          body-false (#'llm-providers.gemini/prepare-body
                      {:messages [{:role "user" :content "hello"}]
                       :web-search false})
          body-tools-only (#'llm-providers.gemini/prepare-body
                           {:messages [{:role "user" :content "hello"}]
                            :tools [{:full-name "search_files"
                                     :description "Search files"
                                     :parameters {:type "object"}}]
                            :web-search false})]
      (is (nil? (:tools body-nil)))
      (is (nil? (:tools body-false)))
      (is (= [{:functionDeclarations [{:name "search_files"
                                       :description "Search files"
                                       :parameters {:type "object"}}]}]
             (:tools body-tools-only))))))

(deftest base-request-proxied-test
  (testing "expands {model} in URL, passes x-goog-api-key header, and extracts output-text"
    (let [req* (atom nil)]
      (with-client-proxied {}
        (fn [req]
          (reset! req* req)
          {:status 200
           :body {:candidates [{:content {:parts [{:text "ok"}]}}]}})
        (let [res (#'llm-providers.gemini/base-request!
                   {:rid "123"
                    :model "gemini-2.5-flash"
                    :api-url "http://localhost:1"
                    :url-relative-path "/v1beta/models/{model}:generateContent"
                    :api-key "test-api-key"
                    :body {:contents []}
                    :stream? false})]
          (is (= "/v1beta/models/gemini-2.5-flash:generateContent" (:uri @req*)))
          (is (= "test-api-key" (get-in @req* [:headers "x-goog-api-key"])))
          (is (= "application/json" (get-in @req* [:headers "Content-Type"])))
          (is (= {:output-text "ok"} res)))))))

(deftest base-request-no-api-key-test
  (testing "omits x-goog-api-key header when api-key is nil"
    (let [req* (atom nil)]
      (with-client-proxied {}
        (fn [req]
          (reset! req* req)
          {:status 200
           :body {:candidates [{:content {:parts [{:text "ok"}]}}]}})
        (#'llm-providers.gemini/base-request!
         {:rid "123"
          :model "gemini-2.5-flash"
          :api-url "http://localhost:1"
          :url-relative-path "/v1beta/models/{model}:generateContent"
          :api-key nil
          :body {:contents []}
          :stream? false})
        (is (nil? (get-in @req* [:headers "x-goog-api-key"])))))))

(deftest base-request-stream-error-redacts-api-key-test
  (testing "redacts api-key from non-200 error body and logs"
    (let [error* (atom nil)]
      (with-client-proxied {}
        (fn [_req]
          {:status 403
           :body "API key secret-12345 is invalid"})
        (#'llm-providers.gemini/base-request!
         {:rid "123"
          :model "gemini-2.5-flash"
          :api-url "http://localhost:1"
          :api-key "secret-12345"
          :body {:contents []}
          :stream? true
          :on-error (fn [err] (reset! error* err))})
        (is (not (string/includes? (:body @error*) "secret-12345")))
        (is (string/includes? (:body @error*) "[REDACTED]"))
        (is (not (string/includes? (:message @error*) "secret-12345")))
        (is (string/includes? (:message @error*) "[REDACTED]"))))))

(deftest base-request-resolves-dynamic-extra-headers-test
  (testing "supports fn-based and map-based extra-headers"
    (let [req* (atom nil)]
      (with-client-proxied {}
        (fn [req]
          (reset! req* req)
          {:status 200
           :body {:candidates [{:content {:parts [{:text "ok"}]}}]}})
        (#'llm-providers.gemini/base-request!
         {:rid "123"
          :model "gemini-2.5-flash"
          :api-url "http://localhost:1"
          :api-key "key"
          :body {:contents [{:role "user" :parts [{:text "hi"}]}]}
          :stream? false
          :extra-headers (fn [{:keys [body]}]
                           {"X-Dynamic" (str (count (:contents body)))})})
        (is (= "1" (get-in @req* [:headers "X-Dynamic"])))))))

(deftest chat!-streaming-text-test
  (testing "streaming SSE text chunks, usage reporting, finish with STOP"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Hello "}] :role "model"}}]}
                  {:candidates [{:content {:parts [{:text "world!"}] :role "model"}
                                :finishReason "STOP"}]
                   :usageMetadata {:promptTokenCount 5 :candidatesTokenCount 2}})]
      (with-redefs [http/post (fn [_url opts]
                                (is (= :stream (:as opts)))
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-key "key"}
         (collecting-callbacks events*)))
      (is (= [[:msg {:type :text :text "Hello "}]
              [:msg {:type :text :text "world!"}]
              [:usage {:input-tokens 5 :output-tokens 2}]
              [:msg {:type :finish :finish-reason "STOP"}]]
             @events*)))))

(deftest chat!-streaming-cached-tokens-test
  (testing "camelCase cachedContentTokenCount is parsed as input-cache-read-tokens and subtracted from input-tokens"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Hello"}] :role "model"}
                                :finishReason "STOP"}]
                   :usageMetadata {:promptTokenCount 100
                                   :candidatesTokenCount 20
                                   :cachedContentTokenCount 80}})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-key "key"}
         (collecting-callbacks events*)))
      (is (= [[:msg {:type :text :text "Hello"}]
              [:usage {:input-tokens 20 :output-tokens 20 :input-cache-read-tokens 80}]
              [:msg {:type :finish :finish-reason "STOP"}]]
             @events*))))

  (testing "snake_case cached_content_token_count is parsed as input-cache-read-tokens and subtracted from input-tokens"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Hello"}] :role "model"}
                                :finish_reason "STOP"}]
                   :usage_metadata {:prompt_token_count 100
                                    :candidates_token_count 20
                                    :cached_content_token_count 75}})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-key "key"}
         (collecting-callbacks events*)))
      (is (= [[:msg {:type :text :text "Hello"}]
              [:usage {:input-tokens 25 :output-tokens 20 :input-cache-read-tokens 75}]
              [:msg {:type :finish :finish-reason "STOP"}]]
             @events*)))))

(deftest chat!-streaming-max-tokens-test
  (testing "finishReason MAX_TOKENS emits limit-reached"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Truncated..."}] :role "model"}
                                :finishReason "MAX_TOKENS"}]
                   :usageMetadata {:promptTokenCount 100 :candidatesTokenCount 32000}})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "generate"}]
          :api-key "key"}
         (collecting-callbacks events*)))
      (is (match? [[:msg {:type :text :text "Truncated..."}]
                   [:usage {:input-tokens 100 :output-tokens 32000}]
                   [:msg {:type :limit-reached :tokens {:promptTokenCount 100 :candidatesTokenCount 32000}}]]
                  @events*)))))

(deftest chat!-streaming-premature-stop-test
  (testing "emits :premature-stop error when stream ends without finishReason"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Incomplete"}] :role "model"}}]})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-key "key"}
         (collecting-callbacks events*)))
      (is (match? [[:msg {:type :text :text "Incomplete"}]
                   [:error {:error/type :premature-stop}]]
                  @events*)))))

(deftest chat!-streaming-cancelled-no-premature-stop-test
  (testing "ignores missing finishReason when cancelled? is true"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:text "Incomplete"}] :role "model"}}]})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-key "key"
          :cancelled? (constantly true)}
         (collecting-callbacks events*)))
      (is (= [[:msg {:type :text :text "Incomplete"}]]
             @events*)))))

(deftest chat!-non-streaming-text-test
  (testing "handles non-streaming generateContent JSON response through callbacks"
    (let [events* (atom [])
          resp-body {:candidates [{:content {:parts [{:text "Non-streaming answer"}]
                                             :role "model"}
                                   :finishReason "STOP"}]
                     :usageMetadata {:promptTokenCount 10 :candidatesTokenCount 4}}]
      (with-client-proxied {}
        (fn [_req]
          {:status 200 :body resp-body})
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-url "http://localhost:1"
          :api-key "key"
          :stream? false}
         (collecting-callbacks events*)))
      (is (= [[:msg {:type :text :text "Non-streaming answer"}]
              [:usage {:input-tokens 10 :output-tokens 4}]
              [:msg {:type :finish :finish-reason "STOP"}]]
             @events*)))))

(deftest chat!-non-streaming-sync-prompt-test
  (testing "returns {:output-text ...} when callbacks are nil"
    (let [resp-body {:candidates [{:content {:parts [{:text "Full sync answer"}]
                                             :role "model"}
                                   :finishReason "STOP"}]}]
      (with-client-proxied {}
        (fn [_req]
          {:status 200 :body resp-body})
        (let [result (llm-providers.gemini/chat!
                      {:model "gemini-2.5-flash"
                       :user-messages [{:role "user" :content "hi"}]
                       :api-url "http://localhost:1"
                       :api-key "key"}
                      nil)]
          (is (= {:output-text "Full sync answer"} result)))))))

(deftest chat!-thinking-test
  (testing "thought parts trigger :on-reason (:started, :thinking, :finished) capturing thoughtSignature as :external-id"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:thought true
                                                    :text "Let me think about this..."
                                                    :thoughtSignature "sig-abc"}]
                                           :role "model"}}]}
                  {:candidates [{:content {:parts [{:text "Here is the answer"}]
                                           :role "model"}
                                :finishReason "STOP"}]
                   :usageMetadata {:promptTokenCount 10 :candidatesTokenCount 10}})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "think"}]
          :api-key "key"
          :reason? true}
         (collecting-callbacks events*)))
      (is (match? [[:reason {:status :started :id string?}]
                   [:reason {:status :thinking :text "Let me think about this..." :id string?}]
                   [:reason {:status :finished :external-id "sig-abc" :id string?}]
                   [:msg {:type :text :text "Here is the answer"}]
                   [:usage {:input-tokens 10 :output-tokens 10}]
                   [:msg {:type :finish :finish-reason "STOP"}]]
                  @events*)))))

(deftest chat!-thinking-then-tool-call-test
  (testing "thinking then tool call chunk properly finishes reasoning before tool call preparation"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:thought true
                                                    :text "I should search files."
                                                    :thoughtSignature "sig-thought-456"}]
                                           :role "model"}}]}
                  {:candidates [{:content {:parts [{:functionCall {:name "search"
                                                                   :args {:query "clojure"}}}]
                                           :role "model"}
                                :finishReason "STOP"}]})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (let [callbacks (assoc (collecting-callbacks events*)
                               :on-tools-called (fn [calls]
                                                  (swap! events* conj [:tools-called calls])
                                                  nil))]
          (llm-providers.gemini/chat!
           {:model "gemini-2.5-flash"
            :user-messages [{:role "user" :content "find clojure"}]
            :api-key "key"
            :reason? true}
           callbacks)))
      (is (match? [[:reason {:status :started :id string?}]
                   [:reason {:status :thinking :text "I should search files." :id string?}]
                   [:reason {:status :finished :external-id "sig-thought-456" :id string?}]
                   [:prepare {:full-name "search" :arguments-text "{\"query\":\"clojure\"}" :id string?}]
                   [:tools-called [{:full-name "search" :arguments {"query" "clojure"} :id string?}]]]
                  @events*)))))

(deftest chat!-parallel-tool-calls-test
  (testing "collects functionCall parts, emits :on-prepare-tool-call, and calls :on-tools-called (with recursion when :new-messages returned)"
    (let [events* (atom [])
          request-count* (atom 0)
          stream-1 (sse-stream
                    {:candidates [{:content {:parts [{:functionCall {:name "get_weather"
                                                                     :args {:location "Paris"}}}
                                                     {:functionCall {:name "get_time"
                                                                     :args {:timezone "UTC"}}}]
                                            :role "model"}
                                  :finishReason "STOP"}]
                     :usageMetadata {:promptTokenCount 12 :candidatesTokenCount 8}})
          stream-2 (sse-stream
                    {:candidates [{:content {:parts [{:text "The weather in Paris is sunny."}]
                                            :role "model"}
                                  :finishReason "STOP"}]
                     :usageMetadata {:promptTokenCount 25 :candidatesTokenCount 6}})]
      (with-redefs [http/post (fn [_url _opts]
                                (let [cnt (swap! request-count* inc)]
                                  {:status 200 :body (if (= cnt 1) stream-1 stream-2)}))]
        (let [callbacks (assoc (collecting-callbacks events*)
                               :on-tools-called
                               (fn [tool-calls]
                                 (swap! events* conj [:tools-called tool-calls])
                                 {:new-messages [{:role "user" :content "sunny"}]
                                  :tools [{:full-name "get_weather"}]}))]
          (llm-providers.gemini/chat!
           {:model "gemini-2.5-flash"
            :user-messages [{:role "user" :content "what's the weather?"}]
            :api-key "key"}
           callbacks)))
      (is (= 2 @request-count*))
      (is (match? [[:prepare {:full-name "get_weather" :arguments-text "{\"location\":\"Paris\"}" :id string?}]
                   [:prepare {:full-name "get_time" :arguments-text "{\"timezone\":\"UTC\"}" :id string?}]
                   [:usage {:input-tokens 12 :output-tokens 8}]
                   [:tools-called [{:full-name "get_weather" :arguments {"location" "Paris"} :id string?}
                                   {:full-name "get_time" :arguments {"timezone" "UTC"} :id string?}]]
                   [:msg {:type :text :text "The weather in Paris is sunny."}]
                   [:usage {:input-tokens 25 :output-tokens 6}]
                   [:msg {:type :finish :finish-reason "STOP"}]]
                  @events*)))))

(deftest chat!-function-call-thought-signature-test
  (testing "tests that thoughtSignature is attached to tool-call map"
    (let [events* (atom [])
          stream (sse-stream
                  {:candidates [{:content {:parts [{:functionCall {:name "search"
                                                                   :args {:q "clojure"}}
                                                    :thoughtSignature "sig-tool-123"}]
                                           :role "model"}
                                 :finishReason "STOP"}]})]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (let [callbacks (assoc (collecting-callbacks events*)
                               :on-tools-called
                               (fn [tool-calls]
                                 (swap! events* conj [:tools-called tool-calls])
                                 nil))]
          (llm-providers.gemini/chat!
           {:model "gemini-2.5-flash"
            :user-messages [{:role "user" :content "search"}]
            :api-key "key"}
           callbacks)))
      (is (match? [[:prepare {:full-name "search" :id string?}]
                   [:tools-called [{:full-name "search"
                                    :arguments {"q" "clojure"}
                                    :thought-signature "sig-tool-123"
                                    :id string?}]]]
                  @events*)))))

(deftest chat!-extra-headers-and-payload-test
  (testing "forwards extraHeaders and deep-merges extraPayload"
    (let [req* (atom nil)]
      (with-client-proxied {}
        (fn [req]
          (reset! req* req)
          {:status 200
           :body {:candidates [{:content {:parts [{:text "done"}] :role "model"}
                                :finishReason "STOP"}]}})
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "hi"}]
          :api-url "http://localhost:1"
          :api-key "key"
          :stream? false
          :extra-headers {"X-Custom-Header" "custom-val"}
          :extra-payload {:generationConfig {:temperature 0.2}}}
         (collecting-callbacks (atom []))))
      (is (= "custom-val" (get-in @req* [:headers "X-Custom-Header"])))
      (is (= 0.2 (get-in @req* [:body :generationConfig :temperature]))))))

(deftest chat!-streaming-grounding-metadata-test
  (testing "streaming chunks with groundingMetadata emit web search start, deduplicated url citations, and finished on STOP"
    (let [events* (atom [])
          chunk-1 {:candidates [{:content {:parts [{:text "Searching..."}] :role "model"}
                                 :groundingMetadata {:webSearchQueries ["clojure news"]}}]}
          chunk-2 {:candidates [{:content {:parts [{:text "Here are updates."}] :role "model"}
                                 :groundingMetadata {:groundingChunks [{:web {:uri "https://clojure.org" :title "Clojure"}}
                                                                       {:web {:uri "https://clojure.org" :title "Clojure"}}
                                                                       {:web {:uri "https://news.ycombinator.com" :title "HN"}}]}}]}
          chunk-3 {:candidates [{:finishReason "STOP"
                                 :groundingMetadata {:webSearchQueries ["clojure news"]
                                                     :groundingChunks [{:web {:uri "https://clojure.org" :title "Clojure"}}
                                                                       {:web {:uri "https://news.ycombinator.com" :title "HN"}}]}}]
                   :usageMetadata {:promptTokenCount 10 :candidatesTokenCount 15}}
          stream (sse-stream chunk-1 chunk-2 chunk-3)]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "clojure news"}]
          :api-key "key"
          :web-search true}
         (collecting-callbacks events*)))
      (is (match? [[:web-search {:status :started
                                 :id string?
                                 :name "web_search"
                                 :input {:query "clojure news"}}]
                   [:msg {:type :text :text "Searching..."}]
                   [:msg {:type :url :title "Clojure" :url "https://clojure.org"}]
                   [:msg {:type :url :title "HN" :url "https://news.ycombinator.com"}]
                   [:msg {:type :text :text "Here are updates."}]
                   [:usage {:input-tokens 10 :output-tokens 15}]
                   [:web-search {:status :finished
                                 :id string?
                                 :output [{:title "Clojure" :url "https://clojure.org"}
                                          {:title "HN" :url "https://news.ycombinator.com"}]
                                 :raw-content {:webSearchQueries ["clojure news"]
                                               :groundingChunks [{:web {:uri "https://clojure.org" :title "Clojure"}}
                                                                 {:web {:uri "https://news.ycombinator.com" :title "HN"}}]}}]
                   [:msg {:type :finish :finish-reason "STOP"}]]
                  @events*))
      (let [started-ev (first (filter #(= :started (get-in % [1 :status])) @events*))
            finished-ev (first (filter #(= :finished (get-in % [1 :status])) @events*))]
        (is (some? (get-in started-ev [1 :id])))
        (is (= (get-in started-ev [1 :id]) (get-in finished-ev [1 :id])))))))

(deftest chat!-streaming-grounding-metadata-snake-case-test
  (testing "snake_case grounding_metadata, web_search_queries, and grounding_chunks"
    (let [events* (atom [])
          chunk-1 {:candidates [{:content {:parts [{:text "Searching..."}] :role "model"}
                                 :grounding_metadata {:web_search_queries ["clojure news"]}}]}
          chunk-2 {:candidates [{:content {:parts [{:text "Found news"}] :role "model"}
                                 :grounding_metadata {:grounding_chunks [{:web {:uri "https://clojure.org" :title "Clojure"}}]}}]}
          chunk-3 {:candidates [{:finish_reason "STOP"
                                 :grounding_metadata {:web_search_queries ["clojure news"]
                                                      :grounding_chunks [{:web {:uri "https://clojure.org" :title "Clojure"}}]}}]}
          stream (sse-stream chunk-1 chunk-2 chunk-3)]
      (with-redefs [http/post (fn [_url _opts]
                                {:status 200 :body stream})]
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "clojure news"}]
          :api-key "key"
          :web-search true}
         (collecting-callbacks events*)))
      (is (match? [[:web-search {:status :started
                                 :id string?
                                 :name "web_search"
                                 :input {:query "clojure news"}}]
                   [:msg {:type :text :text "Searching..."}]
                   [:msg {:type :url :title "Clojure" :url "https://clojure.org"}]
                   [:msg {:type :text :text "Found news"}]
                   [:web-search {:status :finished
                                 :id string?
                                 :output [{:title "Clojure" :url "https://clojure.org"}]
                                 :raw-content {:web_search_queries ["clojure news"]
                                               :grounding_chunks [{:web {:uri "https://clojure.org" :title "Clojure"}}]}}]
                   [:msg {:type :finish :finish-reason "STOP"}]]
                  @events*)))))

(deftest chat!-non-streaming-grounding-metadata-test
  (testing "non-streaming response containing groundingMetadata emits search events and citations"
    (let [events* (atom [])
          resp-body {:candidates [{:content {:parts [{:text "Clojure is great."}] :role "model"}
                                   :finishReason "STOP"
                                   :groundingMetadata {:webSearchQueries ["clojure news"]
                                                       :groundingChunks [{:web {:uri "https://clojure.org" :title "Clojure"}}]}}]
                     :usageMetadata {:promptTokenCount 10 :candidatesTokenCount 5}}]
      (with-client-proxied {}
        (fn [_req]
          {:status 200 :body resp-body})
        (llm-providers.gemini/chat!
         {:model "gemini-2.5-flash"
          :user-messages [{:role "user" :content "clojure news"}]
          :api-url "http://localhost:1"
          :api-key "key"
          :stream? false
          :web-search true}
         (collecting-callbacks events*)))
      (is (match? [[:web-search {:status :started
                                 :id string?
                                 :name "web_search"
                                 :input {:query "clojure news"}}]
                   [:msg {:type :url :title "Clojure" :url "https://clojure.org"}]
                   [:msg {:type :text :text "Clojure is great."}]
                   [:usage {:input-tokens 10 :output-tokens 5}]
                   [:web-search {:status :finished
                                 :id string?
                                 :output [{:title "Clojure" :url "https://clojure.org"}]
                                 :raw-content {:webSearchQueries ["clojure news"]
                                               :groundingChunks [{:web {:uri "https://clojure.org" :title "Clojure"}}]}}]
                   [:msg {:type :finish :finish-reason "STOP"}]]
                  @events*)))))
