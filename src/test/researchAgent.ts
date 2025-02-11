import { HumanMessage } from "@langchain/core/messages"
import { researchChain } from "../agents/researchAgent"

const streamResults = researchChain.stream({
    messages: [new HumanMessage("Explore Europe's Demographics, Including Population, Age Distribution, and More.")]
})

for await (const output of await streamResults) {
    console.log("--- STRAT ---")
    console.log(output)
    console.log("--- END ---")
}