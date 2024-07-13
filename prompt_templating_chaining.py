## Prompt Templating and Chaining
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

OPENAI_API_KEY = ""

template = "You are a business consultant for new companies. What is a good name for a {company} that produces {product}?"

prompt = ChatPromptTemplate.from_template(template)
model = OpenAI(temperature=0.9,
             api_key=OPENAI_API_KEY)

chain = prompt | model | StrOutputParser()

print(chain.invoke({'company':"ABC Startup", 'product':"robots"}))