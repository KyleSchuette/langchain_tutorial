## Storing and Retrieving Chat History
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict, messages_from_dict

OPENAI_API_KEY = ""

history = ChatMessageHistory()
history.add_user_message("Hello! Let's talk about owls")
history.add_ai_message("I'm ready to discuss this animal")
dicts = messages_to_dict(history.messages)
# print(dicts)
new_messages = messages_from_dict(dicts)

llm = OpenAI(temperature=0,
             api_key=OPENAI_API_KEY)
history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(llm=llm, memory=buffer, verbose=True)

print(conversation.invoke(input="What are they?")['response'])

# Access the conversation history
# print(conversation.memory)