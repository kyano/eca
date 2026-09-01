(ns integration.chat.gemini-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [integration.eca :as eca]
   [integration.fixture :as fixture]
   [integration.helper :refer [match-content]]
   [llm-mock.mocks :as llm.mocks]
   [llm-mock.server :as llm-mock.server]
   [matcher-combinators.matchers :as m]
   [matcher-combinators.test :refer [match?]]))

(eca/clean-after-test)

(deftest simple-text
  (eca/start-process!)

  (eca/request! (fixture/initialize-request
                 {:initializationOptions
                  (merge fixture/default-init-options
                         {:defaultModel "my-provider/geminipro"
                          :providers
                          {"myProvider"
                           {:api "gemini"
                            :url (str "http://localhost:" llm-mock.server/port "/gemini")
                            :requiresAuth false
                            :models {"geminipro" {}}}}})
                  :capabilities {:codeAssistant {:chat {}}}}))

  (eca/notify! (fixture/initialized-notification))
  (testing "We use the default model from custom provider"
    (is (match?
         {:chat {:models (m/embeds ["my-provider/geminipro"])
                 :selectModel "my-provider/geminipro"}}
         (eca/client-awaits-server-notification :config/updated))))

  (testing "We send a simple hello message"
    (llm.mocks/set-case! :simple-text-0)
    (let [resp (eca/request! (fixture/chat-prompt-request
                              {:model "my-provider/geminipro"
                               :message "Tell me a joke!"}))
          chat-id (:chatId resp)]

      (is (match?
           {:chatId (m/pred string?)
            :model "my-provider/geminipro"
            :status "prompting"}
           resp))

      (match-content chat-id "user" {:type "text" :text "Tell me a joke!\n"})
      (match-content chat-id "system" {:type "metadata" :title "Some Cool Title"})
      (match-content chat-id "system" {:type "progress" :state "running" :text "Waiting model"})
      (match-content chat-id "system" {:type "progress" :state "running" :text "Generating"})
      (match-content chat-id "assistant" {:type "text" :text "Knock"})
      (match-content chat-id "assistant" {:type "text" :text " knock!"})
      (match-content chat-id "system" {:type "usage"
                                       :sessionTokens 30
                                       :lastMessageCost m/absent
                                       :sessionCost m/absent})
      (match-content chat-id "system" {:type "progress" :state "finished"})
      (is (match?
           {:contents [{:role "user" :parts [{:text "Tell me a joke!"}]}]}
           (llm.mocks/get-req-body :simple-text-0))))))

(deftest cached-text-with-cost
  (eca/start-process!)

  (eca/request! (fixture/initialize-request
                 {:initializationOptions
                  (merge fixture/default-init-options
                         {:defaultModel "my-provider/geminicached"
                          :providers
                          {"myProvider"
                           {:api "gemini"
                            :url (str "http://localhost:" llm-mock.server/port "/gemini")
                            :requiresAuth false
                            :models {"geminicached" {:cost {:input 10000.0
                                                            :output 20000.0
                                                            :cacheRead 1000.0}}}}}})
                  :capabilities {:codeAssistant {:chat {}}}}))

  (eca/notify! (fixture/initialized-notification))
  (testing "We use the default model from custom provider"
    (is (match?
         {:chat {:models (m/embeds ["my-provider/geminicached"])
                 :selectModel "my-provider/geminicached"}}
         (eca/client-awaits-server-notification :config/updated))))

  (testing "We send a simple message with cached prompt and cost"
    (llm.mocks/set-case! :cached-text-0)
    (let [resp (eca/request! (fixture/chat-prompt-request
                              {:model "my-provider/geminicached"
                               :message "Tell me a joke!"}))
          chat-id (:chatId resp)]

      (is (match?
           {:chatId (m/pred string?)
            :model "my-provider/geminicached"
            :status "prompting"}
           resp))

      (match-content chat-id "user" {:type "text" :text "Tell me a joke!\n"})
      (match-content chat-id "system" {:type "metadata" :title "Some Cool Title"})
      (match-content chat-id "system" {:type "progress" :state "running" :text "Waiting model"})
      (match-content chat-id "system" {:type "progress" :state "running" :text "Generating"})
      (match-content chat-id "assistant" {:type "text" :text "Cached response"})
      (match-content chat-id "system" {:type "usage"
                                       :sessionTokens 120
                                       :lastMessageCost "0.68"
                                       :sessionCost "0.68"})
      (match-content chat-id "system" {:type "progress" :state "finished"}))))
