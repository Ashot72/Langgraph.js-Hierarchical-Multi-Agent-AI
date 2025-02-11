
import { Annotation, END, START, StateGraph } from "@langchain/langgraph"
import { BaseMessage, HumanMessage } from "@langchain/core/messages"
import * as readline from "readline"
import { superAgent } from "./utils"
import { RunnableLambda } from "@langchain/core/runnables"
import { researchChain } from "./agents/researchAgent"
import { docChain } from "./agents/docAgent"
import { chartChain } from "./agents/chartAgent"

const State = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "ResearchTeam"
    }),
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Resolve the user's request"
    }),
    reasoning: Annotation<string>({
        reducer: (x, y) => y ?? x
    })
})

const supervisorNode = await superAgent(["ResearchTeam", "DocWritingTeam", "ChartGeneratingTeam"])

const getMessages = RunnableLambda.from((state: typeof State.State) => ({
    messages: state.messages
}))

const joinGraph = RunnableLambda.from((response: any) => ({
    messages: [response.messages[response.messages.length - 1]]
}))

const superGraph = new StateGraph(State)
    .addNode("ResearchTeam", getMessages.pipe(researchChain).pipe(joinGraph))
    .addNode("DocWritingTeam", getMessages.pipe(docChain).pipe(joinGraph))
    .addNode("ChartGeneratingTeam", getMessages.pipe(chartChain).pipe(joinGraph))
    .addNode("supervisor", supervisorNode)
    .addEdge("ResearchTeam", "supervisor")
    .addEdge("DocWritingTeam", "supervisor")
    .addEdge("ChartGeneratingTeam", "supervisor")
    .addConditionalEdges("supervisor", x => x.next, {
        DocWritingTeam: "DocWritingTeam",
        ResearchTeam: "ResearchTeam",
        ChartGeneratingTeam: "ChartGeneratingTeam",
        FINISH: END
    })
    .addEdge(START, "supervisor")

const graph = superGraph.compile();

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
})

const question = "Please ask a question that can be documented and plotted.\n";

const answer: string = await new Promise((resolve) => {
    rl.question(question, (answer) => {
        rl.close()
        resolve(answer)
    })
})

const streamResults = graph.stream({
    messages: [new HumanMessage(answer)]
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

/* Questions

   Please retrieve the popular products of the Lightning Tools company (including their names and descriptions), write a summary of them, and then plot a pie chart.

   How do the American people view Trump's work attitude after his election? Please write a document that outlines the various groups who support or oppose it, and then create a bar chart to visualize the data.

*/