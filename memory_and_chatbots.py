## Memory and Chat Bots
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY = ""

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
# Verbose to False for cleaner UI
conversation = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory())

# print(conversation.run(input="It's a beautiful day!"))

# Chatbot loop
user_input = str()
print("***Welcome to your AI chatbot! Enter a prompt or 'exit' to leave***")
while True:
    user_input = input("User: ")
    if user_input.strip().lower() == 'exit':
        break
    response = conversation.invoke(input=user_input)['response']
    print(f"AI: {response}")
print("Goodbye!")