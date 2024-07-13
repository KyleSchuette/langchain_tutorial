## Plan and Execute Agents
from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain

OPENAI_API_KEY = ""
SERPAPI_API_KEY = ""

# prompt = "Where are the next summer olympics going to be hosted? What is the population of that country raised to the power of 0.43?"
prompt = "Who is the CEO of Tesla? What is the networth of that person? What is the square root of that networth amount?"

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
wikipedia = WikipediaAPIWrapper()

# List of tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to look up facts and statistics."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math."
    ),
]

model = ChatOpenAI(temperature=0,
                   api_key=OPENAI_API_KEY)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.invoke(prompt)