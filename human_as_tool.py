## Human as a Tool
from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

OPENAI_API_KEY = ""

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
tools = load_tools(["human"])

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Causes the agent to ask user for input to answer the prompt
agent_chain.invoke("What is my friend's favorite color?")