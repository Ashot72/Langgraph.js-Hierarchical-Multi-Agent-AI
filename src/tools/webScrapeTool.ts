import colors from "colors";
import { tool } from "@langchain/core/tools";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { z } from "zod";

const scrapeWebpageTool = tool(async (input: { url: string }) => {
    console.log(colors.red("Web Scraping Tool"))

    const loader = new CheerioWebBaseLoader(input.url)
    const docs = await loader.load()
    const formattedDocs = docs.map(
        doc => `<Document name="${doc.metadata?.title}">\n${doc.pageContent}\n</Document>`
    )

    return formattedDocs.join("\n\n")
}, {
    name: "scrape_webpage",
    description: "Scrape the contents of a webpage",
    schema: z.object({
        url: z.string(),
    })
})

export { scrapeWebpageTool }