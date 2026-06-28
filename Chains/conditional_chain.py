from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


model = ChatOllama(model = 'gemma3:4b')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positve or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template= """You are a customer support representative.

Customer feedback:
{feedback}

Write exactly ONE short professional reply to the customer.

Do not explain your answer.
Do not give multiple options.
Only return the reply.
""",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="""
You are a customer support representative.

Customer feedback:
{feedback}

Write exactly ONE short professional apology and offer assistance.

Do not explain.
Do not give multiple options.
Only return the reply.
""" ,
    input_variables=['feedback']
)



branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive' ,prompt2 |model | parser ),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser ),
    RunnableLambda(lambda x : 'Could not find sentiment')
)


chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': "this is a terrible phone"})
print(result)