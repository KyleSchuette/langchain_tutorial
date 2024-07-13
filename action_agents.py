## Action Agents
from langchain_openai import OpenAI
from langchain.agents import  initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.agent_toolkits.load_tools import get_all_tool_names
import pprint

OPENAI_API_KEY = ""

prompt = "When was the 3rd President of the United States born? What is that year raised to the power of 3?"

# List available tools
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(get_all_tool_names())

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.invoke(prompt)