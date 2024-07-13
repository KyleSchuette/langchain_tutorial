## Simple Sequential Chains
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

OPENAI_API_KEY = ""

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
template = "What is a good name for a company that makes {product}?"

first_prompt = PromptTemplate.from_template(template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

# Test first chain
# print(first_chain.invoke({'product':"colorful socks"}))

second_template = "Write a catchphrase for the following company: {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# Output of first chain is used as input to second chain
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)

catchphrase = overall_chain.invoke("colorful socks")['output']
print(catchphrase)