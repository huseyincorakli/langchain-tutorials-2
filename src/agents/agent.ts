import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import * as dotenv from "dotenv";


dotenv.config();
const agent = async () => {
    //tavily : bir arama aracı
    const agentTools = [new TavilySearchResults({ maxResults: 3,
        callbacks: [{
            handleToolStart: async (tool) => {
                console.log("🔍 Arama başlatıldı...");
            },
            handleToolEnd: async (output) => {
                console.log("✅ Arama tamamlandı");
                console.log("Arama sonuçları:", output);
            },
            handleToolError: async (error) => {
                console.error("❌ Arama sırasında hata:", error);
            }
        }]
     })];

    const agentModel = new ChatOpenAI({
        modelName: "google/gemini-flash-1.5-8b-exp",
        temperature: 0,
        configuration: {
            baseURL: "https://openrouter.ai/api/v1",
        },
        maxTokens: 1000,
    });

    const agentCheckpointer = new MemorySaver();

    const agent = createReactAgent({
        llm: agentModel,
        tools: agentTools,
        checkpointSaver: agentCheckpointer,
    });

    const agentFinalState = await agent.invoke(
        { messages: [new HumanMessage("what is the current weather in Kahramanmaras")] },
        { configurable: { thread_id: "42" } },
    );

    console.log(
        agentFinalState.messages[agentFinalState.messages.length - 1].content,
    );

    const agentNextState = await agent.invoke(
        { messages: [new HumanMessage("who is current president of Kahramanmaras?")] },
        { configurable: { thread_id: "42" } },
      );
      
      console.log(
        agentNextState.messages[agentNextState.messages.length - 1].content,
      );

      const graph = await agent.getGraphAsync();
      const mermaid  = graph.drawMermaid()
      console.log(mermaid);
      
}

export default agent;