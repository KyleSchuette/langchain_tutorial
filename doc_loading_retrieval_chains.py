## Document Loading and QA Retrieval
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain

OPENAI_API_KEY = ""

loader = TextLoader("./LangChain Tutorial/state_of_the_union_2022.txt", encoding='UTF-8')
documents = loader.load()
# print(documents)

# Split into smaller chunks, 3 Splitting Recursions: \n\n, \n, " "
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
store = Chroma.from_documents(texts, embeddings, collection_name="state-of-the-union")

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

# print(chain.invoke("Did Biden mention Covid?"))
print(chain.invoke("What did Biden say about the war in Ukraine?")['result'])