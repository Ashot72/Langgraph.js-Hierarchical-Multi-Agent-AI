# Hierarchical Multi-Agent AI in LangGraph.js
In AI, hierarchical multi-agent systems refer to structured arrangements where multiple agents work together under a layered or tiered system. 
The Supervisor (or Controller) oversees and coordinates the lower-level agents to ensure efficient task execution. In an app we can do this by
composing different subgraphs and creating a top-level supervisor agent, along with mid-level agents.

In this Hierarchical Multi-Agent system, we created three independent agents, where the supervisor can redirect to each of them using the 'next' role. This demonstrates that the app can be composed of not just one, but multiple independent agents working together. The first agent researches the given question, the second generates a document and saves it into the database, and the third generates a chart (using Chart.js) and sends an email to the admin using different tools. Note that the supervisor may decide to call the agents (as well as the tools) multiple times and in different orders until the task is completed. Since the data content and chart information are stored in the database, a web dashboard can be created to analyze the data. However, in this app, I only email the result along with the chart. You can run each agent separately to understand it better.

For example, I asked the AI for the popular products of the [Lightning Tools](https://lightningtools.com/) company. The AI then saved the response in the database, generated a chart, and emailed it to the admin.

[LangGraph.js](https://langchain-ai.github.io/langgraphjs/) is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits; cycles, controllability, and persistence.

![graph image](https://github.com/Ashot72/Langgraph.js-Hierarchical-Multi-Agent-AI/blob/main/multiAgent.png)

As a cloud-based database, [MongoDB](https://www.mongodb.com/atlas/database) is used with [Prisma ORM](https://www.prisma.io/ ). 

[Stripe](https://stripe.com/) payment service (test mode) is integrated into the app, allowing you to make purchases with test card numbers and view all transactions on Stripe's *Payments* panel.

We add tracing in [LangSmith](https://www.langchain.com/langsmith) to monitor model performance, trace execution flows, and evaluate LLM interactions, ensuring they operate efficiently and meet expectations.


To get started.
```
       # Clone the repository

         git clone https://github.com/Ashot72/Langgraph.js-Hierarchical-Multi-Agent-AI
         cd Langgraph.js-Hierarchical-Multi-Agent-AI

       # Create the .env file based on the env.example.txt file and include the respective keys.
       
       # installs dependencies
         npm install

       # to start
         npm start
         
       # to start researcher agent only
         npm run start-research
      
       # to start doc generator agent only
         npm run start-doc
         
       # to start chart agent only
         npm run start-chart
```

Go to [Hierarchical Multi-Agent AI in LangGraph.js Video](https://youtu.be/uoLzeRC3-_8) page 

Go to [Hierarchical Multi-Agent AI in LangGraph.js Description](https://ashot72.github.io/Langgraph.js-Hierarchical-Multi-Agent-AI/doc.html) page
