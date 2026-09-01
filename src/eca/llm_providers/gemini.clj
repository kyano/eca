(ns eca.llm-providers.gemini
  (:require
   [clojure.java.io :as io]
   [clojure.set :as set]
   [clojure.string :as string]
   [clojure.walk :as walk]
   [cheshire.core :as json]
   [hato.client :as http]
   [eca.client-http :as client]
   [eca.llm-util :as llm-util]
   [eca.logger :as logger]
   [eca.message-sanitize :as message-sanitize]
   [eca.shared :refer [assoc-some deep-merge join-api-url]]))

(set! *warn-on-reflection* true)

(def ^:private logger-tag "[GEMINI]")

(def ^:private default-api-url "https://generativelanguage.googleapis.com")
(def ^:private default-stream-path "/v1beta/models/{model}:streamGenerateContent?alt=sse")
(def ^:private default-non-stream-path "/v1beta/models/{model}:generateContent")

(def ^:private gemini-unsupported-schema-keys
  #{:additionalProperties :$schema :exclusiveMinimum :exclusiveMaximum})

(defn ^:private normalize-nullable-type
  [m]
  (if (vector? (:type m))
    (let [nullable? (contains? (set (:type m)) "null")
          types (remove #{"null"} (:type m))]
      (if (seq types)
        (cond-> (assoc m :type (first types))
          nullable? (assoc :nullable true))
        (dissoc m :type)))
    m))

(def ^:private schema-too-deep-fallback
  {:type "object" :properties {}})

(defn ^:private sanitize-schema-for-gemini
  [schema]
  (try
    (walk/postwalk (fn [x]
                     (if (map? x)
                       (-> (apply dissoc x gemini-unsupported-schema-keys)
                           normalize-nullable-type)
                       x))
                   schema)
    (catch StackOverflowError _
      (logger/warn logger-tag "Tool parameter schema too deeply nested to sanitize for Gemini; using an empty schema for this tool")
      schema-too-deep-fallback)))

(defn ^:private extract-parts
  [content supports-image?]
  (cond
    (string? content)
    (if (string/blank? content)
      []
      [{:text content}])

    (sequential? content)
    (vec
     (keep (fn [block]
             (when-let [t (:type block)]
               (case (name t)
                 "text" (when-not (string/blank? (:text block))
                          {:text (:text block)})
                 "image" (when supports-image?
                           {:inlineData {:mimeType (or (:media-type block) "image/png")
                                         :data (:base64 block)}})
                 nil)))
           content))

    :else
    (let [s (str content)]
      (if (string/blank? s)
        []
        [{:text s}]))))

(defn ^:private drop-unpaired-tool-messages
  "Drops tool_call entries without matching tool_call_output, and
   tool_call_output entries without matching tool_call."
  [messages]
  (let [call-ids (into #{}
                       (keep #(when (= "tool_call" (:role %))
                                (get-in % [:content :id])))
                       messages)
        output-ids (into #{}
                         (keep #(when (= "tool_call_output" (:role %))
                                  (get-in % [:content :id])))
                         messages)
        paired-ids (set/intersection call-ids output-ids)]
    (filterv (fn [{:keys [role content]}]
               (case role
                 "tool_call" (contains? paired-ids (:id content))
                 "tool_call_output" (contains? paired-ids (:id content))
                 true))
             messages)))

(defn ^:private reorder-tool-turn
  [turn]
  (let [{calls "tool_call" outputs "tool_call_output"} (group-by :role turn)
        sorted-calls (sort-by #(get-in % [:content :index] Long/MAX_VALUE) calls)
        call-id->pos (into {} (map-indexed (fn [i m] [(get-in m [:content :id]) i]) sorted-calls))]
    (concat sorted-calls (sort-by #(call-id->pos (get-in % [:content :id]) Long/MAX_VALUE) outputs))))

(defn ^:private group-parallel-tool-calls
  [messages]
  (let [tool-msg? #(contains? #{"tool_call" "tool_call_output"} (:role %))
        call-id->name (into {} (keep (fn [m]
                                       (when (= "tool_call" (:role m))
                                         (when-let [id (get-in m [:content :id])]
                                           (when-let [n (or (get-in m [:content :full-name])
                                                            (get-in m [:content :name]))]
                                             [id n]))))
                                     messages))]
    (->> messages
         (mapv (fn [m]
                 (if (and (= "tool_call_output" (:role m))
                          (nil? (get-in m [:content :full-name])))
                   (if-let [n (call-id->name (get-in m [:content :id]))]
                     (assoc-in m [:content :full-name] n)
                     m)
                   m)))
         (partition-by tool-msg?)
         (mapcat (fn [group]
                   (if-not (tool-msg? (first group))
                     group
                     (let [turn-id #(get-in % [:content :tool-turn-id])]
                       (->> (if (some turn-id group)
                              (partition-by turn-id group)
                              [group])
                            (mapcat reorder-tool-turn)))))))))

(defn ^:private sort-user-turn-parts
  "Within a user turn, non-functionResponse parts (text, image) must come
   before functionResponse parts. Gemini 3.x rejects a user turn where
   functionResponse is followed by text with:
   'Requests ending with a model turn are not supported.'"
  [parts]
  (vec (sort-by (fn [part] (if (:functionResponse part) 1 0)) parts)))

(defn ^:private merge-adjacent-roles
  [messages]
  (reduce (fn [acc msg]
            (let [last-msg (peek acc)]
              (if (and last-msg (= (:role last-msg) (:role msg)))
                (let [merged-parts (into (:parts last-msg) (:parts msg))
                      final-parts (if (= "user" (:role msg))
                                    (sort-user-turn-parts merged-parts)
                                    merged-parts)]
                  (conj (pop acc) (assoc last-msg :parts final-parts)))
                (conj acc (if (= "user" (:role msg))
                            (update msg :parts sort-user-turn-parts)
                            msg)))))
          []
          messages))

(defn ^:private normalize-messages
  ([messages]
   (normalize-messages messages true))
  ([messages supports-image?]
   (->> (group-parallel-tool-calls messages)
        (keep (fn [{:keys [role content]}]
                (let [foreign-api? (let [origin (:api content)]
                                     (and origin (not= :gemini origin)))]
                  (case role
                    "user" {:role "user" :parts (extract-parts content supports-image?)}
                    "assistant" {:role "model" :parts (extract-parts content supports-image?)}
                    "system" nil

                    "tool_call"
                    (when-not foreign-api?
                      (let [sig (or (:thought-signature content) (:external-id content))]
                        {:role "model"
                         :parts [(assoc-some {:functionCall {:name (:full-name content)
                                                             :args (or (:arguments content) {})}}
                                             :thoughtSignature sig)]}))

                    "tool_call_output"
                    (when-not foreign-api?
                      {:role "user"
                       :parts [{:functionResponse
                                (assoc-some {:response {:result (llm-util/stringfy-tool-result content)}}
                                            :name (:full-name content))}]})

                    "reason"
                    (when-not foreign-api?
                      {:role "model"
                       :parts [(assoc-some {:text (or (:text content) "") :thought true}
                                           :thoughtSignature (or (:external-id content) (:thought-signature content)))]})

                    nil))))
        merge-adjacent-roles)))

(defn ^:private ->tools [tools]
  (mapv (fn [tool]
          {:name (:full-name tool)
           :description (:description tool)
           :parameters (sanitize-schema-for-gemini (:parameters tool))})
        tools))

(defn ^:private message-text
  [content]
  (cond
    (string? content) content
    (sequential? content) (->> content (keep :text) (string/join "\n"))
    :else nil))

(defn ^:private extract-system-instruction
  [messages instructions]
  (let [system-texts (->> messages
                          (filter #(= "system" (:role %)))
                          (keep (comp message-text :content)))
        combined (string/join "\n" (remove string/blank? (concat system-texts [instructions])))]
    (when-not (string/blank? combined)
      {:parts [{:text combined}]})))

(defn ^:private build-body [{:keys [contents system-instruction tools web-search web_search reason? max-output-tokens]}]
  (let [web-search? (or web-search web_search)
        built-tools (cond-> []
                      (seq tools) (conj {:functionDeclarations (->tools tools)})
                      web-search? (into [{:googleSearch {}} {:urlContext {}}]))]
    (assoc-some
     {:contents contents
      :generationConfig (cond-> {:maxOutputTokens (or max-output-tokens 32000)}
                          reason? (assoc :thinkingConfig {:includeThoughts true}))}
     :systemInstruction system-instruction
     :tools (when (seq built-tools) built-tools))))

(defn ^:private finalize-gemini-contents
  "Ensures contents adheres to Gemini API turn requirements:
   1. Drops turns with empty parts.
   2. Drops leading model turns (conversations must begin with a user turn).
   3. Ensures the request does not end on a model turn (Gemini strictly rejects
      requests ending with a model turn with HTTP 400: 'Requests ending with a
      model turn are not supported.').
   4. Ensures contents is never empty (Gemini requires at least one user turn)."
  [contents]
  (let [non-empty (filterv (comp seq :parts) contents)
        without-leading-model (vec (drop-while #(= "model" (:role %)) non-empty))]
    (cond
      (empty? without-leading-model)
      [{:role "user" :parts [{:text "Hello"}]}]

      (= "model" (:role (peek without-leading-model)))
      (conj without-leading-model {:role "user" :parts [{:text "Continue"}]})

      :else
      without-leading-model)))

(defn ^:private prepare-body
  [{:keys [messages past-messages user-messages instructions supports-image? extra-payload extraPayload] :as opts}]
  (let [all-messages (or messages (vec (concat past-messages user-messages)))
        sanitized-messages (drop-unpaired-tool-messages all-messages)
        contents (-> (normalize-messages sanitized-messages (if (some? supports-image?) supports-image? true))
                     finalize-gemini-contents
                     vec)
        system-instruction (extract-system-instruction all-messages instructions)
        base-body (build-body (assoc opts :contents contents :system-instruction system-instruction))
        payload (or extra-payload extraPayload)]
    (if (seq payload)
      (deep-merge base-body payload)
      base-body)))

(defn ^:private finish-reasoning! [state* on-reason]
  (let [{:keys [started? finished? id thought-signature]} (:reasoning @state*)]
    (when (and started? (not finished?))
      (when on-reason
        (on-reason (assoc-some {:status :finished :id id}
                               :external-id thought-signature)))
      (swap! state* assoc-in [:reasoning :finished?] true))))

(defn ^:private finish-search! [state* on-server-web-search]
  (let [{:keys [started? finished? id outputs raw-content]} (:search @state*)]
    (when (and started? (not finished?))
      (when on-server-web-search
        (on-server-web-search {:status :finished
                               :id id
                               :output outputs
                               :raw-content raw-content}))
      (swap! state* assoc-in [:search :finished?] true))))

(defn ^:private handle-grounding!
  [candidate state* {:keys [on-message-received on-server-web-search]}]
  (when-let [gm (or (:groundingMetadata candidate) (:grounding_metadata candidate))]
    (swap! state* assoc-in [:search :raw-content] gm)
    (let [queries (or (:webSearchQueries gm) (:web_search_queries gm))]
      (when (and (seq queries)
                 (not (get-in @state* [:search :started?])))
        (when on-server-web-search
          (on-server-web-search {:status :started
                                 :id (get-in @state* [:search :id])
                                 :name "web_search"
                                 :input {:query (first queries)}}))
        (swap! state* assoc-in [:search :started?] true)))
    (let [chunks (or (:groundingChunks gm) (:grounding_chunks gm))]
      (doseq [chunk chunks]
        (let [web (or (:web chunk) chunk)
              uri (or (:uri web) (:url web))
              title (:title web)]
          (when (and uri (not (contains? (get-in @state* [:search :seen-uris]) uri)))
            (swap! state* (fn [st]
                            (-> st
                                (update-in [:search :seen-uris] conj uri)
                                (update-in [:search :outputs] conj {:title title :url uri}))))
            (when on-message-received
              (on-message-received {:type :url :title title :url uri}))))))))

(defn ^:private handle-parts!
  [parts state* {:keys [on-message-received on-reason on-prepare-tool-call]}]
  (doseq [part parts]
    (cond
      (or (:functionCall part) (:function_call part))
      (let [fc (or (:functionCall part) (:function_call part))
            call-id (or (:id fc) (:id part) (str (random-uuid)))
            fc-name (some-> (or (:name fc) (:full-name fc)) name)
            fc-args (or (:args fc) (:arguments fc) {})
            args-text (if (string? fc-args) fc-args (json/generate-string fc-args))
            args-map (if (string? fc-args)
                       (try (json/parse-string fc-args) (catch Exception _ fc-args))
                       (walk/stringify-keys fc-args))
            raw-sig (or (:thoughtSignature part) (:thought_signature part)
                        (:thoughtSignature fc) (:thought_signature fc))
            sig (or raw-sig
                    (when (empty? (:tool-calls @state*))
                      (get-in @state* [:reasoning :thought-signature])))
            tool-call (assoc-some {:id call-id
                                   :full-name fc-name
                                   :arguments args-map}
                                  :thought-signature sig
                                  :external-id sig)]
        (finish-reasoning! state* on-reason)
        (when on-prepare-tool-call
          (on-prepare-tool-call {:id call-id
                                 :full-name fc-name
                                 :arguments-text args-text}))
        (swap! state* update :tool-calls conj tool-call))

      (or (:thought part) (:thoughtSignature part) (:thought_signature part))
      (let [sig (or (:thoughtSignature part) (:thought_signature part))]
        (when sig
          (swap! state* assoc-in [:reasoning :thought-signature] sig))
        (let [{:keys [started? id]} (:reasoning @state*)]
          (when-not started?
            (when on-reason (on-reason {:status :started :id id}))
            (swap! state* assoc-in [:reasoning :started?] true))
          (when-let [text (:text part)]
            (when (seq text)
              (when on-reason (on-reason {:status :thinking :id id :text text}))))))

      (:text part)
      (let [text (:text part)]
        (finish-reasoning! state* on-reason)
        (when (and on-message-received (seq text))
          (on-message-received {:type :text :text text}))))))

(defn ^:private handle-stream
  [data state* {:keys [on-message-received on-reason on-tools-called on-usage-updated on-server-web-search] :as callbacks} recur-fn]
  (let [candidate (first (:candidates data))
        parts (get-in candidate [:content :parts])
        finish-reason (or (:finishReason candidate) (:finish_reason candidate))
        finish-reason (when finish-reason (if (keyword? finish-reason) (name finish-reason) (str finish-reason)))
        usage (or (:usageMetadata data) (:usage_metadata data))]
    (handle-grounding! candidate state* callbacks)
    (handle-parts! parts state* callbacks)
    (when usage
      (swap! state* assoc :last-usage usage)
      (when on-usage-updated
        (let [prompt-tokens (or (:promptTokenCount usage) (:prompt_token_count usage) 0)
              output-tokens (or (:candidatesTokenCount usage) (:candidates_token_count usage) 0)
              cached-tokens (or (:cachedContentTokenCount usage) (:cached_content_token_count usage))]
          (on-usage-updated
           (assoc-some {:input-tokens (if cached-tokens
                                        (- prompt-tokens cached-tokens)
                                        prompt-tokens)
                        :output-tokens output-tokens}
                       :input-cache-read-tokens cached-tokens)))))
    (when finish-reason
      (swap! state* assoc :has-finish-reason? true)
      (finish-reasoning! state* on-reason)
      (finish-search! state* on-server-web-search)
      (if (seq (:tool-calls @state*))
        (let [tool-turn-id (str (random-uuid))
              calls (mapv #(assoc % :tool-turn-id tool-turn-id) (:tool-calls @state*))]
          (swap! state* assoc :tool-calls [])
          (when on-tools-called
            (when-let [{:keys [new-messages tools fresh-api-key provider-auth]} (on-tools-called calls)]
              (recur-fn new-messages tools (or fresh-api-key (:api-key provider-auth))))))
        (when on-message-received
          (if (= "MAX_TOKENS" finish-reason)
            (on-message-received {:type :limit-reached :tokens (:last-usage @state*)})
            (on-message-received {:type :finish :finish-reason finish-reason})))))))

(defn ^:private base-request!
  [{:keys [rid model body api-url api-key url-relative-path stream? http-client extra-headers
           cancelled? stream-idle-timeout-seconds on-error on-stream]}]
  (let [stream? (if (some? stream?) stream? (boolean on-stream))
        default-path (if stream? default-stream-path default-non-stream-path)
        rel-path (or url-relative-path default-path)
        expanded-path (if (and model (string/includes? rel-path "{model}"))
                        (llm-util/expand-model-placeholder rel-path model)
                        rel-path)
        url (join-api-url (or api-url default-api-url) expanded-path)
        extra-headers (if (fn? extra-headers)
                        (extra-headers {:body body})
                        extra-headers)
        headers (client/merge-llm-headers
                 (cond-> {"Content-Type" "application/json"}
                   (not (string/blank? api-key)) (assoc "x-goog-api-key" api-key)
                   extra-headers (merge extra-headers)))
        response* (atom nil)
        on-error (or on-error identity)]
    (llm-util/log-request logger-tag rid url body headers)
    (try
      (let [{:keys [status body] resp-headers :headers}
            (http/post url
                       {:headers headers
                        :body (json/generate-string body)
                        :throw-exceptions? false
                        :decompress-body false
                        :http-client (client/merge-with-global-http-client http-client)
                        :as (if stream? :stream :json)})]
        (if (not= 200 status)
          (let [body-str (if stream? (slurp body) (if (string? body) body (json/generate-string body)))
                sanitized-body (if (and (not (string/blank? api-key)) (string? body-str))
                                 (string/replace body-str api-key "[REDACTED]")
                                 body-str)]
            (logger/warn logger-tag "Unexpected response status: %s body: %s" status sanitized-body)
            (reset! response*
                    (on-error {:message (format "Gemini response status: %s body: %s" status sanitized-body)
                               :status status
                               :body sanitized-body
                               :headers resp-headers})))
          (if stream?
            (let [{:keys [touch-fn set-reading-fn stop-fn reason*]}
                  (llm-util/start-stream-watchdog!
                   body cancelled?
                   (when stream-idle-timeout-seconds
                     {:idle-timeout-ms (* 1000 stream-idle-timeout-seconds)}))]
              (try
                (with-open [rdr (io/reader body)]
                  (doseq [[event data] (llm-util/event-data-seq rdr)]
                    (set-reading-fn false)
                    (touch-fn)
                    (llm-util/log-response logger-tag rid event data)
                    (when on-stream
                      (on-stream event data))
                    (set-reading-fn true)))
                (catch clojure.lang.ExceptionInfo e
                  (if (= :premature-stop (:error/type (ex-data e)))
                    (do
                      (logger/warn logger-tag "Stream ended with empty response, retrying")
                      (on-error (merge {:message (ex-message e)} (ex-data e))))
                    (throw e)))
                (catch java.io.IOException e
                  (let [reason @reason*]
                    (cond
                      (= :cancelled reason)
                      (throw (ex-info "Stream cancelled" {:silent? true}))

                      (= :idle-timeout reason)
                      (on-error {:message (format "Stream idle timeout: no data received for %d seconds"
                                                  (or stream-idle-timeout-seconds 120))
                                 :exception e})

                      :else
                      (on-error {:exception e
                                 :message (llm-util/connection-error-message e)}))))
                (finally
                  (stop-fn))))
            (do
              (llm-util/log-response logger-tag rid "response" body)
              (if on-stream
                (do
                  (on-stream nil body)
                  (reset! response* body))
                (reset! response* {:output-text (->> (:candidates body)
                                                     first
                                                     :content
                                                     :parts
                                                     (keep :text)
                                                     (string/join ""))}))))))
      (catch Exception e
        (reset! response*
                (on-error {:exception e
                           :message (llm-util/connection-error-message e)}))))
    @response*))

(defn chat!
  "Executes a chat request against Gemini API, supporting both streaming (SSE)
   and non-streaming modes."
  ([opts]
   (chat! opts nil))
  ([{:keys [cancelled?] :as opts}
    {:keys [on-error] :as callbacks}]
   (let [body (prepare-body opts)]
     (if-not callbacks
       ;; Sync mode (non-streaming)
       (base-request!
        (assoc opts
               :rid (llm-util/gen-rid)
               :body body
               :stream? false))
       ;; Callback mode (streaming or non-streaming)
       (let [stream? (if (some? (:stream? opts))
                       (:stream? opts)
                       (let [path (or (:url-relative-path opts) "")]
                         (or (string/blank? path)
                             (string/includes? path "streamGenerateContent")
                             (string/includes? path "alt=sse"))))
             state* (atom {:tool-calls []
                           :reasoning {:started? false
                                       :finished? false
                                       :id (str (random-uuid))
                                       :thought-signature nil}
                           :search {:started? false
                                    :finished? false
                                    :id (str (random-uuid))
                                    :seen-uris #{}
                                    :outputs []
                                    :raw-content nil}
                           :has-finish-reason? false
                           :last-usage nil})]
         (letfn [(execute-request! [current-opts current-body]
                   (let [recur-fn (fn [new-messages tools fresh-key]
                                    (let [new-messages (message-sanitize/sanitize-outbound-messages new-messages)
                                          recur-opts (cond-> (assoc current-opts :messages new-messages :tools tools)
                                                       fresh-key (assoc :api-key fresh-key))
                                          recur-body (prepare-body recur-opts)]
                                      (reset! state* {:tool-calls []
                                                      :reasoning {:started? false
                                                                  :finished? false
                                                                  :id (str (random-uuid))
                                                                  :thought-signature nil}
                                                      :search {:started? false
                                                               :finished? false
                                                               :id (str (random-uuid))
                                                               :seen-uris #{}
                                                               :outputs []
                                                               :raw-content nil}
                                                      :has-finish-reason? false
                                                      :last-usage nil})
                                      (execute-request! recur-opts recur-body)))]
                     (base-request!
                      (assoc current-opts
                             :rid (llm-util/gen-rid)
                             :body current-body
                             :stream? stream?
                             :on-error on-error
                             :on-stream (fn [_event data]
                                          (handle-stream data state* callbacks recur-fn))))
                     (when stream?
                       (finish-reasoning! state* (:on-reason callbacks))
                       (finish-search! state* (:on-server-web-search callbacks))
                       (when (seq (:tool-calls @state*))
                         (let [tool-turn-id (str (random-uuid))
                               calls (mapv #(assoc % :tool-turn-id tool-turn-id) (:tool-calls @state*))]
                           (swap! state* assoc :tool-calls [])
                           (when-let [on-tools-called (:on-tools-called callbacks)]
                             (when-let [{:keys [new-messages tools fresh-api-key provider-auth]} (on-tools-called calls)]
                               (recur-fn new-messages tools (or fresh-api-key (:api-key provider-auth)))))))
                       (when-not (or (:has-finish-reason? @state*) (and cancelled? (cancelled?)))
                         (logger/warn logger-tag "Stream ended without finishReason, retrying")
                         (when on-error
                           (on-error {:message "Stream ended without completion signal"
                                      :error/type :premature-stop}))))))]
           (execute-request! opts body)))))))
