(ns llm-mock.gemini
  (:require
   [cheshire.core :as json]
   [clojure.string :as string]
   [llm-mock.mocks :as llm.mocks]
   [org.httpkit.server :as hk]))

(defn ^:private sse-send!
  "Send one unnamed `data: {...}` SSE chunk, mirroring Gemini's
  `streamGenerateContent?alt=sse` wire format (no `event:` line), followed by
  a blank line. Keeps the connection open until the scenario finishes."
  [ch m]
  (hk/send! ch (str "data: " (json/generate-string m) "\n\n") false))

;; Simple text cases
(defn ^:private simple-text-0 [ch]
  (sse-send! ch {:candidates [{:content {:parts [{:text "Knock"}]}}]})
  (sse-send! ch {:candidates [{:content {:parts [{:text " knock!"}]}
                               :finishReason "STOP"}]
                 :usageMetadata {:promptTokenCount 10
                                 :candidatesTokenCount 20}})
  (hk/close ch))

(defn ^:private cached-text-0 [ch]
  (sse-send! ch {:candidates [{:content {:parts [{:text "Cached response"}]}
                               :finishReason "STOP"}]
                 :usageMetadata {:promptTokenCount 100
                                 :candidatesTokenCount 20
                                 :cachedContentTokenCount 80}})
  (hk/close ch))

(defn ^:private chat-title-text-0
  "Title generation goes through the same endpoint, but ECA calls it
  synchronously (no streaming callbacks), so Gemini requests it non-streaming
  and parses the response as a single plain JSON body, not an SSE stream."
  [ch]
  (hk/send! ch
            {:status 200
             :headers {"Content-Type" "application/json"}
             :body (json/generate-string
                    {:candidates [{:content {:parts [{:text "Some Cool Title"}]}
                                   :finishReason "STOP"}]})}
            true))

(defn ^:private title-generation-request?
  "Gemini carries the system instruction under `:systemInstruction`, unlike
  Anthropic's `:system` array — detect a title-generation request there."
  [body]
  (string/includes? (str (get-in body [:systemInstruction :parts])) llm.mocks/chat-title-generator-str))

(defn handle-gemini-generate-content [req]
  (let [body (some-> (slurp (:body req))
                     (json/parse-string true))]
    (if (title-generation-request? body)
      (hk/as-channel req {:on-open chat-title-text-0})
      (hk/as-channel
       req
       {:on-open (fn [ch]
                   (hk/send! ch {:status 200
                                 :headers {"Content-Type" "text/event-stream; charset=utf-8"
                                           "Cache-Control" "no-cache"
                                           "Connection" "keep-alive"}}
                             false)
                   (llm.mocks/set-req-body! llm.mocks/*case* body)
                   (case llm.mocks/*case*
                     :simple-text-0 (simple-text-0 ch)
                     :cached-text-0 (cached-text-0 ch)))}))))
