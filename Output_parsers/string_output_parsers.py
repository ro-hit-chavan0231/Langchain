from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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



parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)