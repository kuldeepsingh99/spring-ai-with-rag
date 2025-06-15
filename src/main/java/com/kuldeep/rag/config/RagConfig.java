package com.kuldeep.rag.config;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
public class RagConfig {

    private final Logger logger = LoggerFactory.getLogger(RagConfig.class);


    @Autowired
    VectorStore vectorStore;

    private static final String MARKER_PREFIX = "__MARKER__:";

    @PostConstruct
    public void loadAllTxtFiles() {
        try {
            PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
            Resource[] resources = resolver.getResources("classpath:data/*.txt");

            for (Resource resource : resources) {
                String filename = Objects.requireNonNull(resource.getFilename());
                String tag = filename.replace(".txt", "");

                if (isTagAlreadyLoaded(tag)) {
                    logger.info("[" + tag + "] already loaded. Skipping.");
                    continue;
                }

                // 1. Read file and convert to documents
                TextReader textReader = new TextReader(resource);
                List<Document> rawDocs = textReader.get();

                // 2. Split documents into chunks
                List<Document> splitDocs = new TokenTextSplitter().apply(rawDocs);

                // 3. Add metadata tag to each doc
                List<Document> taggedDocs = splitDocs.stream()
                        .map(doc -> new Document(doc.getText(), Map.of("source", tag)))
                        .collect(Collectors.toList());

                // 4. Add a marker doc to identify this file was loaded
                taggedDocs.add(new Document(MARKER_PREFIX + tag, Map.of("source", tag, "type", "marker")));

                // 5. Save to vector store
                vectorStore.add(taggedDocs);
                logger.info("[" + tag + "] Loaded " + taggedDocs.size() + " documents.");
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to load data files", e);
        }
    }

    private boolean isTagAlreadyLoaded(String tag) {
        try {
            List<Document> results = vectorStore.similaritySearch(SearchRequest.builder()
                    .query(MARKER_PREFIX + tag)
                    .topK(1)
                    .build());

            return results.stream()
                    .anyMatch(doc -> doc.getText().equalsIgnoreCase(MARKER_PREFIX + tag));
        } catch (Exception e) {
            logger.error("Error checking if tag already loaded [" + tag + "]: " + e.getMessage());
            return false;
        }
    }
}
