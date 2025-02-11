import colors from "colors";
import fs from "fs"
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { Chart, registerables, ChartItem, ChartTypeRegistry } from "chart.js/auto"
import { createCanvas } from 'canvas';
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import { prisma } from "../db";

Chart.register(...registerables);

const chartTool = tool(async ({ type, title, data }, config: LangGraphRunnableConfig) => {
    const thread_id: number = config.configurable?.thread_id

    console.log(colors.red("Chart Tool"), thread_id || "test")

    const labels = data.map(d => d.label);
    const values = data.map(d => d.value);
    const barColors = [
        "#b91d47", // Red
        "#00aba9", // Teal
        "#2b5797", // Blue
        "#e8c3b9", // Light Pink
        "#1e7145", // Dark Green
        "#ff5733", // Orange
        "#900c3f", // Maroon
        "#581845", // Purple
        "#ffd700", // Gold
        "#4caf50", // Green
        "#ff9800", // Deep Orange
        "#2196f3", // Bright Blue
        "#673ab7", // Deep Purple
        "#ffeb3b"  // Yellow
    ];

    const width = 400;
    const height = 400;
    const canvas = createCanvas(width, height);

    new Chart(canvas as unknown as ChartItem, {
        type: type as keyof ChartTypeRegistry,
        data: {
            labels,
            datasets: [{
                backgroundColor: barColors,
                data: values
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });

    const base64Image = canvas.toDataURL('image/png');

    //for tesing no thread_id to escape the db update
    if (thread_id) {
        await prisma.output.update({
            where: { thread_id },
            data: { chart: base64Image },
        });
    } else {
        fs.writeFileSync('chart.png', Buffer.from(base64Image.split("base64,")[1], "base64"))
    }

    return base64Image;
}, {
    name: "generate_chart",
    description: "Generates a chart (e.g., bar, line, pie) matching ChartTypeRegistry from an array of data points using Chart.js, displaying the chart for the user. Each data point includes a label (e.g., 'January', 'Sales') and a corresponding numeric value (e.g., 100, 250).",
    schema: z.object({
        title: z.string().describe("The title to be displayed at the top of the generated chart. This should provide context for the data presented, such as 'Sales Performance for 2023' or 'Monthly Revenue'."),
        type: z.enum(['bar', 'line', 'pie', 'scatter', 'bubble', 'doughnut', 'polarArea', 'radar']).describe("The type of chart to generate. Choose from 'bar', 'line', 'pie', 'scatter', 'bubble', 'doughnut', 'polarArea', or 'radar'."),
        data: z.object({
            label: z.string().describe("A descriptive label for the data point (e.g., 'January', 'Sales')."),
            value: z.number().describe("The numeric value corresponding to the label (e.g., 100, 250).")
        })
            .array()
    })
})

export default chartTool