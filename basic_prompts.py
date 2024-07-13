## Prompts and LLMs
from langchain_openai import OpenAI

OPENAI_API_KEY = ""

# Temperature lies between 0 and 1
# Default OpenAI model is currently GPT-3.5-turbo-instruct
llm = OpenAI(temperature=0.9,
             api_key=OPENAI_API_KEY)
## Show model in use
print(llm.model_name)

prompt = "What would a good company name be for a company that makes colorful socks?"

## Single Response
# response = llm.invoke(prompt)
# print(response)

## Multiple Responses
result = llm.generate([prompt]*5)
for company_name in result.generations:
    print(company_name[0].text)