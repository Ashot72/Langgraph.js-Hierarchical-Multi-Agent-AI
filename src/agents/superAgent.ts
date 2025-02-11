import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";
import { z } from "zod"
import { llm } from "../utils";

async function createSuperAgent(systemPrompt: string, memebers: string[]): Promise<Runnable> {
    const options = ["FINISH", ...memebers] as const

    const routeTool = {
        name: "route",
        descrription: "Select the next role.",
        schema: z.object({
            reasoning: z.string().describe("Explain the decision behind selecting the next role."),
            next: z.enum(options),
            instructions: z.string().describe("The specific instructions of the sub-task the next role should be accomplish.")
        })
    }

    let prompt = ChatPromptTemplate.fromMessages([
        ["system", systemPrompt],
        new MessagesPlaceholder("messages"),
        ["system", "Given the conversation above, who should act next? Or should we FINISH? Select one of {options}"]
    ]);

    prompt = await prompt.partial({
        options: options.join(", "),
        team_members: memebers.join(", ")
    })

    let supervisor = prompt.pipe(llm.bindTools([routeTool], {
        runName: "route"
    }))
        .pipe((x) =>
            (x.tool_calls && x.tool_calls.length > 0)
                ?
                ({
                    reasoning: x.tool_calls[0].args.reasoning,
                    next: x.tool_calls[0].args.next,
                    instructions: x.tool_calls[0].args.instructions,
                })
                :
                ({
                    next: options[0],
                    instructions: ""
                })
        )

    return supervisor
}

export default createSuperAgent
