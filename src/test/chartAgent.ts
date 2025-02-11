import { HumanMessage } from "@langchain/core/messages"
import { chartChain } from "../agents/chartAgent"

const streamResults = chartChain.stream({
    messages: [new HumanMessage("Generate a pie chart for the population of the US, EU, and Asia.")]
})

for await (const output of await streamResults) {
    console.log("--- STRAT ---")
    console.log(output)
    console.log("--- END ---")
}