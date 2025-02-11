import colors from "colors";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { agentStateModifier, llm, superAgent, runAgentNode } from "../utils";
import { scrapeWebpageTool } from "../tools/webScrapeTool";

export const ResearchTeamState = Annotation.Root({
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
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Solve the human's questions"
    }),
    reasoning: Annotation<string>({
        reducer: (x, y) => y ?? x
    })
})

const tavilyTool = new TavilySearchResults();

const searchNode = (state: typeof ResearchTeamState.State) => {
    console.log(colors.magenta("Search Node"))

    const stateModifier = agentStateModifier(
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
        [tavilyTool],
        state.team_members ?? ["Search"]
    )

    const searchAgent = createReactAgent({
        llm,
        tools: [tavilyTool],
        stateModifier
    })
    return runAgentNode({ state, agent: searchAgent, name: "Search" })
}

const scrapeNode = (state: typeof ResearchTeamState.State) => {
    console.log(colors.magenta("Web Scrape Node"))

    const stateModifier = agentStateModifier(
        "You are a research assistant who can scrape specified urls for more detailed information using the scrapeWebpage function.",
        [scrapeWebpageTool],
        state.team_members ?? ["WebScraper"]
    )

    const scrapeAgent = createReactAgent({
        llm,
        tools: [scrapeWebpageTool],
        stateModifier
    })
    return runAgentNode({ state, agent: scrapeAgent, name: "WebScraper" })
}

const supervisorNode = await superAgent(["Search", "WebScraper"])

const researchGraph = new StateGraph(ResearchTeamState)
    .addNode("Search", searchNode)
    .addNode("WebScraper", scrapeNode)
    .addNode("supervisor", supervisorNode)
    .addEdge("Search", "supervisor")
    .addEdge("WebScraper", "supervisor")
    .addConditionalEdges("supervisor", x => x.next, {
        Search: "Search",
        WebScraper: "WebScraper",
        FINISH: END
    })
    .addEdge(START, "supervisor")

export const researchChain = researchGraph.compile()
