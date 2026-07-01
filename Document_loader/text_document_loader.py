from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model = 'tinyllama')


prompt = PromptTemplate(
    template="Write the a sort summary having (2-5 lines) for the following easy \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('football.txt',encoding= 'utf-8')

docs = loader.load()

# print(docs[0].page_content)

# print(len(docs))

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))