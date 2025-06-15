package com.kuldeep.rag.controller;

import org.springframework.ai.chat.client.ChatClient;
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
@RequestMapping("/chat-rental")
public class ChatRentalController {

    private final OllamaChatModel chatModel;
    private final ChatClient chatClient;
    private final VectorStore vectorStore;

    public ChatRentalController(OllamaChatModel ollamaChatModel, VectorStore vectorStore, ChatClient.Builder chatClient) {
        this.vectorStore = vectorStore;
        this.chatModel = ollamaChatModel;
        this.chatClient = chatClient.build();
    }

    @GetMapping()
    public String chatWithRAG(@RequestParam("message") String message) {
        String systemMessageString = """
            You are a helpful assistant who answers questions about the ola Auto Car Rental with Type af car rental Offered with Prices.
            
            * Identify customer queries related to ola Auto Car rental.
            * respond queries related to car rental and prices.
            * Refer to the SERVICE section to provide accurate answers. 
            * If the answer is found, respond clearly and accurately, quoting the relevant section or clause if necessary. 
            * If the information is not available in the ola auto rental Terms & Conditions, respond by directing the customer to send an email to the provided contact for further assistance.

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
}
