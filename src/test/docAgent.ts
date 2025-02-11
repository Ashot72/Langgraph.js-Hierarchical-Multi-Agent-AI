import { HumanMessage } from "@langchain/core/messages"
import { docChain } from "../agents/docAgent"

const streamResults = docChain.stream({
    messages: [new HumanMessage("Write a initial lyrics of a song then improve it by updating the lyrics.")
    ]
}, {
    configurable: {
        thread_id: Math.floor(10000 + Math.random() * 90000)
    }
})

for await (const output of await streamResults) {
    console.log("--- STRAT ---")
    console.log(output)
    console.log("--- END ---")
}