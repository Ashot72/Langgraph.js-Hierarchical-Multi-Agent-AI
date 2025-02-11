import colors from "colors";
import { BaseMessage } from "@langchain/core/messages";
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { RunnableLambda } from "@langchain/core/runnables";
import { agentStateModifier, dbContent, llm, runAgentNode, superAgent } from "../utils";
import { readDocTool, updateDocTool, writeDocTool } from "../tools/dbTool";

const DocumentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y)
    }),
    team_members: Annotation<string[]>({
        reducer: (x, y) => x.concat(y)
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "supervisior"
    }),
    db_id: Annotation<string>({
        reducer: (x, y) => y ?? x
    }),
    current_data: Annotation<string>({
        reducer: (x, y) => (y ? `${x}\n${y}` : x)
    }),
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Solve the humans's questions"
    }),
    reasoning: Annotation<string>({
        reducer: (x, y) => y ?? x
    })
})

const docWritingNode = async (state: typeof DocumentState.State) => {
    console.log(colors.magenta("Document Writing Node"))

    const stateModifier = agentStateModifier(
        `You are an expert writing a research document ONCE using 'write_db' tool.\n${state.current_data}`,
        [writeDocTool],
        state.team_members ?? []
    )

    const docWriterAgent = createReactAgent({
        llm,
        tools: [writeDocTool],
        stateModifier
    })

    const ctxAwareDocWritingAgent = dbContent.pipe(docWriterAgent)
    return runAgentNode({ state, agent: ctxAwareDocWritingAgent, name: "DocWriter" })
}

const docUpdatingNode = async (state: typeof DocumentState.State) => {
    console.log(colors.magenta("Document Updating Node"))

    const stateModifier = agentStateModifier(
        `As an expert senior researcher, and your task is ONLY to read and update the existing document if nessesary using 'read_db' and 'update_db' tools.\n ${state.current_data}`,
        [readDocTool, updateDocTool],
        state.team_members ?? []
    )

    const docUpdatingAgent = createReactAgent({
        llm,
        tools: [readDocTool, updateDocTool],
        stateModifier
    })

    const ctxAwareDocUpdatingAgent = dbContent.pipe(docUpdatingAgent)
    return runAgentNode({ state, agent: ctxAwareDocUpdatingAgent, name: "DocUpdater" })
}

const supervisorNode = await superAgent(["DocWriter", "DocUpdater"])

const documentAgent = new StateGraph(DocumentState)
    .addNode("DocWriter", docWritingNode)
    .addNode("DocUpdater", docUpdatingNode)
    .addNode("supervisor", supervisorNode)
    .addEdge("DocWriter", "supervisor")
    .addEdge("DocUpdater", "supervisor")
    .addConditionalEdges("supervisor", x => x.next, {
        DocWriter: "DocWriter",
        DocUpdater: "DocUpdater",
        FINISH: END
    })
    .addEdge(START, "supervisor")

const enterDocChain = RunnableLambda.from(
    ({ messages }: { messages: BaseMessage[] }) => {
        return {
            messages,
            team_members: ["Document Writer", "Document Updater"]
        }
    }
)

export const docChain = enterDocChain.pipe(documentAgent.compile())
