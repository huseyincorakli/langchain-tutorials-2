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
import { AIMessage, HumanMessage, SystemMessage, trimMessages } from "@langchain/core/messages";

dotenv.config();

const promptTemplate2 = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ],
    ["placeholder", "{messages}"],
  ]);
const llm = new ChatOpenAI({
    modelName: "deepseek/deepseek-chat:free",
    temperature: 0,
    configuration: {
        baseURL: "https://openrouter.ai/api/v1",
    },
    maxTokens: 1000,
});

const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human",
  });
  
  export const messages = [
    new SystemMessage("you're a good assistant"),
    new HumanMessage("hi! I'm bob"),
    new AIMessage("hi!"),
    new HumanMessage("I like vanilla ice cream"),
    new AIMessage("nice"),
    new HumanMessage("whats 2 + 2"),
    new AIMessage("4"),
    new HumanMessage("thanks"),
    new AIMessage("no problem!"),
    new HumanMessage("having fun?"),
    new AIMessage("yes!"),
  ];
  await trimmer.invoke(messages);

  const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
  });

  const callModel4 = async (state: typeof GraphAnnotation.State) => {
    const trimmedMessage = await trimmer.invoke(state.messages);
    const prompt = await promptTemplate2.invoke({
      messages: trimmedMessage,
      language: state.language,
    });
    const response = await llm.invoke(prompt);
    return { messages: [response] };
  };
  
  const workflow4 = new StateGraph(GraphAnnotation)
    .addNode("model", callModel4)
    .addEdge(START, "model")
    .addEdge("model", END);
  
  const app4 = workflow4.compile({ checkpointer: new MemorySaver() });
  export default app4;