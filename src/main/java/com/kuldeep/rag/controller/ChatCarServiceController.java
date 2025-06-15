package com.kuldeep.rag.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/chat")
public class ChatCarServiceController {

    private final OllamaChatModel chatModel;
    private final ChatClient chatClient;
    private final VectorStore vectorStore;

    public ChatCarServiceController(OllamaChatModel ollamaChatModel, VectorStore vectorStore, ChatClient.Builder chatClient) {
        this.vectorStore = vectorStore;
        this.chatModel = ollamaChatModel;
        this.chatClient = chatClient.build();
    }

    @GetMapping("/chat-without-advisor")
    public String chatWithRAG(@RequestParam("message") String message) {
        String systemMessageString = """
            You are a helpful assistant who answers questions about the Car Services with Type af Service Offered wit Prices.
            
            * Identify customer queries related to Wiz Auto Car Service.
            * respond queries related to car services and prices.
            * Refer to the SERVICE section to provide accurate answers. 
            * If the answer is found, respond clearly and accurately, quoting the relevant section or clause if necessary. 
            * If the information is not available in the Car Service Terms & Conditions, respond by directing the customer to send an email to the provided contact for further assistance.

            SERVICE:
            {svc}
            """;

        Message userMessage = new UserMessage(message);

        List<Document> similarDocuments = vectorStore.similaritySearch(message);
        String tncString = similarDocuments.stream()
                .map(Document::getFormattedContent)
                .collect(Collectors.joining("\n"));

        SystemPromptTemplate systemPromptTemplate = new SystemPromptTemplate(systemMessageString);
        Message systemMessage = systemPromptTemplate.createMessage(Map.of("svc", tncString));

        Prompt prompt = new Prompt(List.of(systemMessage, userMessage));

        ChatResponse chatResponse = chatModel.call(prompt);
        return "Response: " + chatResponse.getResult().getOutput().getText();
    }

    @GetMapping("/with-rag-advisor")
    public String chatWithRAGAdvisor(@RequestParam("message") String message) {
        String systemMessageString = """
            You are a helpful assistant who answers questions about the Car Services with Type af Service Offered with Prices.
            
            * Identify customer queries related to Wiz Auto Car Service.
            * respond queries related to car services and prices.
            * Refer to the Car Service Terms & Conditions to provide accurate answers. 
            * If the answer is found in the Terms & Conditions, respond clearly and accurately, quoting the relevant section or clause if necessary. 
            * If the information is not available in the Car Service Terms & Conditions, respond by directing the customer to send an email to the provided contact for further assistance.

            """;

        SystemPromptTemplate systemPromptTemplate = new SystemPromptTemplate(systemMessageString);


        ChatResponse chatResponse = ChatClient.builder(chatModel).build()
                .prompt(systemPromptTemplate.create())
                .advisors(new QuestionAnswerAdvisor(vectorStore))
                .user(message)
                .call()
                .chatResponse();

        return "Response: " + chatResponse.getResult().getOutput().getText();
    }
}
