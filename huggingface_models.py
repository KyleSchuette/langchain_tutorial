## Using Different LLM Models
from langchain import HuggingFaceHub

HUGGINGFACEHUB_API_TOKEN = ""

llm = HuggingFaceHub(repo_id="google/flan-t5-base", 
    model_kwargs={"temperature": 0, "max_length": 64},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

prompt = "What are good fitness tips?"

print(llm.invoke(prompt))