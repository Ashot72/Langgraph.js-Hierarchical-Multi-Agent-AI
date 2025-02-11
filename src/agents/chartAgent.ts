import colors from "colors";
import nodemailer from "nodemailer"
import { prisma } from "../db";
import { BaseMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { llm, agentStateModifier, runAgentNode, superAgent } from "../utils";
import chartTool from "../tools/chartTool";

export const ChartGeneratorState = Annotation.Root({
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
    current_data: Annotation<string>({
        reducer: (x, y) => (y ? `${x}\n${y}` : x)
    }),
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Solve the human's questions"
    }),
    reasoning: Annotation<string>({
        reducer: (x, y) => y ?? x
    })
})

const chartGeneratingNode = async (state: typeof ChartGeneratorState.State) => {
    console.log(colors.magenta("Chart Generating Node"))

    const stateModifier = agentStateModifier(
        `You are a data visulazation expert tasked with generating charts for a research project using 'generate_chart' tool.\n ${state.current_data}`,
        [chartTool],
        ["ChartGenerator"]
    )

    const chartGenerationAgent = createReactAgent({
        llm,
        tools: [chartTool],
        stateModifier
    })

    return runAgentNode({ state, agent: chartGenerationAgent, name: "ChartGenerator" })
}

const emailnotificationNode = async (state: typeof ChartGeneratorState.State, config) => {
    console.log(colors.magenta("Email Sending Node"))

    const thread_id: number = config.configurable?.thread_id

    const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: {
            user: process.env.EMAIL_USER_FROM,
            pass: process.env.EMAIL_PASSWORD,
        }
    });

    // for testing no thread_id
    if (thread_id) {
        const data = await prisma.output.findUnique({
            where: { thread_id }
        });

        if (data?.chart) {
            const lnToBr = (content: string) =>
                content.replace(/(?:\r\n|\r|\n)/g, "<br>");

            const mailOptions = {
                from: process.env.EMAIL_USER_FROM,
                to: process.env.EMAIL_USER_TO,
                subject: process.env.EMAIL_SUBJECT,
                html: `${lnToBr(data?.content)}`,
                attachments: [
                    {
                        filename: 'chart.png',
                        content: data?.chart.split("base64,")[1],
                        encoding: 'base64'
                    }
                ]
            };

            const result: string = await new Promise((resolve, reject) => {
                transporter.sendMail(mailOptions, (error, info) => {
                    if (error) {
                        reject("Error sending email: " + error.message)
                    } else {
                        resolve("Email sent successfully: " + info.response)
                    }
                });
            })

            console.log(colors.green(result))
        } else {
            console.log(colors.red("Chart has not been created, so no email will be sent."))
        }
    }
    return state
}

const supervisorNode = await superAgent(["ChartGenerator"])

const chartGraph = new StateGraph(ChartGeneratorState)
    .addNode("ChartGenerator", chartGeneratingNode)
    .addNode("EmailNotification", emailnotificationNode)
    .addNode("supervisor", supervisorNode)
    .addEdge("ChartGenerator", "supervisor")
    .addConditionalEdges("supervisor", x => x.next, {
        ChartGenerator: "ChartGenerator",
        FINISH: "EmailNotification"
    })
    .addEdge(START, "supervisor")
    .addEdge("EmailNotification", END)

export const chartChain = chartGraph.compile()

