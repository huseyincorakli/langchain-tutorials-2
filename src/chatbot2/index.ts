import { HumanMessage } from "@langchain/core/messages";
import app4, { messages } from "./chatbot.js";

const  chatbot2 = async()=>{
    const config = { configurable: { thread_id: 'unique-conversation-id' } };


    const input1 = {
        messages: [...messages, new HumanMessage("Benim adım ne?")],
        language: "Turkish"
    };
    const input2 = {
        messages: [...messages, new HumanMessage("Sana hangi matematik sorusunu sordum?")],
        language: "Turkish"
    };
    
  
    
    const output1 = await app4.invoke(input1, config);
    console.log(output1.messages[output1.messages.length - 1]);
    

    const output2 = await app4.invoke(input2, config);
    console.log(output2.messages[output2.messages.length - 1]);
  
}

export default chatbot2;