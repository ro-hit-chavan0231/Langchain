from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatOllama(model = "tinyllama")

prompt1 = PromptTemplate(
    template= "Generate a detail report on {topic}",
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template="Generate the 5 pointer summary from the following \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result =  chain.invoke({'topic' :'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()