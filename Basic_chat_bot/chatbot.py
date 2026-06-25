from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


model = ChatOllama(model='phi3:mini', temperature= 0)


chat_history = [
    SystemMessage(content="You are a helpful assistant. Answer only the user's question. Keep your response under 3-5 sentences.Do not continue the conversation by writing the user's next message.")
]

while True:
    user_input =input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result =model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)