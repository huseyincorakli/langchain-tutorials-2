import app3 from "./chatbot.js";

const  chatbot = async()=>{
    const config = { configurable: { thread_id: 'unique-conversation-id' } };
    const config2 = { configurable: { thread_id: 'unique2-conversation-id' } };


    const input1 = {
        messages: [
            {
                role: "user",
                content: "Merhaba, benim adım Ali",
            }
        ],
        language: "Turkish"
    };
    
    // İkinci mesaj (dil belirtmeye gerek yok, hatırlanıyor)
    const input2 = {
        messages: [
            {
                role: "user",
                content: "Benim adım neydi?",
            }
        ]
    };

    const input3= {
        messages: [
            {
                role: "user",
                content: "Benim adım neydi?",
            }
            
        ],
        language:"Latin"
    };
    
    const output1 = await app3.invoke(input1, config);
    console.log(output1.messages[output1.messages.length - 1]);
    
    const output2 = await app3.invoke(input2, config);
    console.log(output2.messages[output2.messages.length - 1]);

    const output3 = await app3.invoke(input3, config2);
    console.log(output3.messages[output3.messages.length - 1]);
}

export default chatbot;