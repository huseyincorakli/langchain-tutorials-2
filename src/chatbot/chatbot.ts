import { ChatOpenAI } from "@langchain/openai";
import {
    START,
    END,
    StateGraph,
    MemorySaver,
    MessagesAnnotation,
    Annotation,
} from "@langchain/langgraph";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";

dotenv.config();

const llm = new ChatOpenAI({
    modelName: "deepseek/deepseek-chat:free",
    temperature: 0,
    configuration: {
        baseURL: "https://openrouter.ai/api/v1",
    },
    maxTokens: 1000,
});

const promptTemplate = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ],
    ["placeholder", "{messages}"],
]);

const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
  });
  
  // Define the function that calls the model
  const callModel3 = async (state: typeof GraphAnnotation.State) => {
    const prompt = await promptTemplate.invoke(state);
    const response = await llm.invoke(prompt);
    return { messages: [response] };
  };

  const workflow3 = new StateGraph(GraphAnnotation)
  .addNode("model", callModel3)
  .addEdge(START, "model")
  .addEdge("model", END);

  const app3 = workflow3.compile({ checkpointer: new MemorySaver() });

  export default app3;