import colors from "colors";
import { tool } from "@langchain/core/tools";
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import { z } from "zod";
import { prisma } from "../db";

const writeDocTool = tool(async ({ content }, config: LangGraphRunnableConfig) => {
    const thread_id: number = config.configurable?.thread_id

    console.log(colors.red("Write Tool"), thread_id)

    await prisma.output.upsert({
        where: { thread_id },
        update: { content },
        create: {
            content,
            thread_id
        }
    });

    return `The content "${content}" has been successfully inserted under thread ID ${thread_id}.`
}, {
    name: "write_db",
    description: "Create and save content to the database",
    schema: z.object({
        content: z.string().describe("The content of the document to be saved in the database.")
    })
})

const updateDocTool = tool(async ({ content }, config: LangGraphRunnableConfig) => {
    const thread_id: number = config.configurable?.thread_id

    console.log(colors.red("Update Tool"), thread_id)

    const updated = await prisma.output.update({
        where: { thread_id },
        data: { content },
    });

    return `The content with thread ID ${thread_id} has been successfully edited and saved: ${updated.content}`

}, {
    name: "update_db",
    description: "Update the document.",
    schema: z.object({
        content: z.string().describe("The content of the document to be saved in the database.")
    })
})

const readDocTool = tool(async (_, config: LangGraphRunnableConfig) => {
    const thread_id: number = config.configurable?.thread_id

    console.log(colors.red("Reading Tool"), thread_id)

    const data = await prisma.output.findUnique({
        where: { thread_id }
    });

    return data?.content || ""
}, {
    name: "read_db",
    description: "Read content from the database"
})

export { writeDocTool, updateDocTool, readDocTool }