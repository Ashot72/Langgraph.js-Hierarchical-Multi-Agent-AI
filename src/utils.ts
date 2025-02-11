import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Runnable, RunnableLambda } from "@langchain/core/runnables";
import { StructuredToolInterface } from "@langchain/core/tools";
import { LangGraphRunnableConfig, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { prisma } from "./db";
import createSuperAgent from "./agents/superAgent";

const llm = new ChatOpenAI({
    modelName: "gpt-4-turbo",
    temperature: 0
});

const superAgent = async (members: string[]) =>
    await createSuperAgent(
        "You are a supervisior tasked with managing a conversation between the" +
        " following workers: {team_members}. Given the following user request," +
        " respond with the worker to act next. Each worker will perform a" +
        " task and respond with their results and status. When finished," +
        " respond with FINISH\n\n" +
        " Select strategically to minimize the number of steps taken.",
        members
    )

const agentStateModifier = (
    systemPrompt: string,
    tools: StructuredToolInterface[],
    team_members: string[],
): ((state: typeof MessagesAnnotation.State) => BaseMessage[]) => {

    const toolNames = tools.map(t => t.name).join(", ")
    const systemMsgStart = new SystemMessage(systemPrompt +
        "\nWork autonmously according to your speciality, using the tools available to you." +
        " Do not ask for clarification." +
        " Your other team members (and other teams) will collaborate with you with their own specification." +
        ` You are chosen for a reason!. You are one of the following team members: ${team_members.join(", ")}.`)
    const systemMsgEnd = new SystemMessage(`Supervisor instructions: ${systemPrompt}\n` +
        `Remember, you individually can only use these tools: ${toolNames}` +
        "\n\nEnd ifyou have already completed requested task. Communicate the work completed."
    )

    return (state: typeof MessagesAnnotation.State): any[] =>
        [systemMsgStart, ...state.messages, systemMsgEnd];
}

async function runAgentNode(params: {
    state: any,
    agent: Runnable,
    name: string
}) {
    const { state, agent, name } = params
    const result = await agent.invoke({
        messages: state.messages,
    })
    const lastMessage = result.messages[result.messages.length - 1]
    return {
        messages: [new HumanMessage({ content: lastMessage.content, name })]
    }
}

const dbContent = new RunnableLambda({
    func: async (state: {
        messages: BaseMessage[]
        next: string
        thread_id: string
        instructions: string
    }, config: LangGraphRunnableConfig) => {
        const data = await prisma.output.findUnique({
            where: { thread_id: config.configurable?.thread_id }
        });
        return { ...state, current_data: data?.content }
    }
})

export { llm, agentStateModifier, runAgentNode, superAgent, dbContent }