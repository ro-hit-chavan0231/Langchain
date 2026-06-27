from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate



model = ChatOllama(
    model = 'tinyllama'
)


# Detailed Report Prompt
template1 = PromptTemplate(
    template = "Write a Detailed report on {topic}",
    input_variables = ['topic']
)



# Summary Prompt
template2 = PromptTemplate(
    template = "Write a 5 ine summary on the following text. /n report on {text}",
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic':'Black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)