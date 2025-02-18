import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";
import * as dotenv from "dotenv";
import { CallbackManager } from "@langchain/core/callbacks/manager";

dotenv.config();

const rag2 = async () => {
    try {
        const embeddings = new OllamaEmbeddings({
            model: "all-minilm",
            baseUrl: "url",
        });

        

        // Initialize the LLM with a more specific system message
        const llm = new ChatOpenAI({
            modelName: "deepseek/deepseek-chat:free",
            temperature: 0.1, // Reduced temperature for more deterministic output
            configuration: {
                baseURL: "https://openrouter.ai/api/v1",
            },
            maxTokens: 1000,
            callbackManager: CallbackManager.fromHandlers({
                handleLLMEnd: (output) => {
                    console.log("Token Usage:", {
                        promptTokens: output.llmOutput?.tokenUsage
                    });
                }
            })
        });

        // Define search schema
        const searchSchema = z.object({
            query: z.string().describe("Search query to run."),
            section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
        });

        // Create structured LLM with a more specific prompt
        const structuredLlm = llm.withStructuredOutput(searchSchema);

        // Define template for final answer generation
        const template = `Use the following pieces of context to answer the question at the end.
                            If you don't know the answer, just say that you don't know, don't try to make up an answer.
                            Use three sentences maximum and keep the answer as concise as possible.
                            Always say "thanks for asking!" at the end of the answer.

                            {context}

                            Question: {question}

                            Helpful Answer:`;

        const promptTemplateCustom = ChatPromptTemplate.fromMessages([
            ["user", template],
        ]);

        // Load and process documents
        const cheerioLoader = new CheerioWebBaseLoader(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            {
                selector: "p",
            }
        );

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 250,
            chunkOverlap: 50
        });

        const docs = await cheerioLoader.load();
        const allSplits = await splitter.splitDocuments(docs);
        
        // Add section metadata
        const totalDocuments = allSplits.length;
        const third = Math.floor(totalDocuments / 3);

        allSplits.forEach((document, i) => {
            if (i < third) {
                document.metadata["section"] = "beginning";
            } else if (i < 2 * third) {
                document.metadata["section"] = "middle";
            } else {
                document.metadata["section"] = "end";
            }
        });

        // Setup memory vector store
        const vectorStoreQA = new MemoryVectorStore(embeddings);
        await vectorStoreQA.addDocuments(allSplits);

        // Define state annotations
        const StateAnnotationQA = Annotation.Root({
            question: Annotation<string>,
            search: Annotation<z.infer<typeof searchSchema>>,
            context: Annotation<Document[]>,
            answer: Annotation<string>,
        });

        // Improved query analysis function with better error handling
        const analyzeQuery = async (state: typeof StateAnnotationQA.State) => {
            const analyzePrompt = `You are a search query analyzer. Your task is to analyze the following question and generate a structured search query.
            Question: "${state.question}"
            Instructions:
            1. Generate a clear search query based on the question
            2. Determine which section (beginning, middle, or end) is most relevant
            3. Return ONLY a JSON object with the following structure, nothing else:
            {
                "query": "your search query",
                "section": "beginning" | "middle" | "end"
            }
            Remember:
            - The section MUST be exactly one of: "beginning", "middle", or "end"
            - Only return the JSON object, no other text
            - Ensure the JSON is properly formatted`;

                        try {
                            const result = await structuredLlm.invoke(analyzePrompt);
                            console.log("Analysis result:", result); // Debug log
                            return { search: result };
                        } catch (error) {
                            console.error("Error in query analysis:", error);
                            // Fallback to a default search if analysis fails
                            return {
                                search: {
                                    query: state.question,
                                    section: "middle"
                                }
                            };
                        }
                    };

        // Improved retrieval function
        const retrieveQA = async (state: typeof StateAnnotationQA.State) => {
            const filter = (doc:Document) => doc.metadata.section === state.search.section;
            const retrievedDocs = await vectorStoreQA.similaritySearch(
              state.search.query,
              2,
              filter
            );
            return { context: retrievedDocs };
          };

        // Improved answer generation function
        const generateQA = async (state: typeof StateAnnotationQA.State) => {
            try {
                const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
                const messages = await promptTemplateCustom.invoke({
                    question: state.question,
                    context: docsContent,
                });
                const response = await llm.invoke(messages);
                return { answer: response.content };
            } catch (error) {
                console.error("Error in answer generation:", error);
                return { 
                    answer: "I apologize, but I encountered an error while generating the answer. Thanks for asking!"
                };
            }
        };

        // Create and compile the graph
        const graphQA = new StateGraph(StateAnnotationQA)
            .addNode("analyzeQuery", analyzeQuery)
            .addNode("retrieveQA", retrieveQA)
            .addNode("generateQA", generateQA)
            .addEdge("__start__", "analyzeQuery")
            .addEdge("analyzeQuery", "retrieveQA")
            .addEdge("retrieveQA", "generateQA")
            .addEdge("generateQA", "__end__")
            .compile();

        // Test the implementation with error handling
        const inputs = { 
            question: "What does the end of the post say about Task Decomposition??" 
        };

        try {
            const stream = await graphQA.stream(inputs, { 
                streamMode: "messages" 
            });

            for await (const [message, _metadata] of stream) {
                process.stdout.write(message.content);
            }
        } catch (error) {
            console.error("Error in streaming:", error);
            console.log("Attempting to get non-streaming response...");
            
            // Fallback to non-streaming response
            const result = await graphQA.invoke(inputs);
            console.log("Final answer:", result.answer);
        }

    } catch (error) {
        console.error("Fatal error in RAG implementation:", error);
        throw error;
    }
};

export default rag2;