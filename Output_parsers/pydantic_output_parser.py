from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import (ResponseSchema,StructuredOutputParser,)



model = ChatOllama(
    model = 'gemma3:4b',
    temperature= 0
)


class person(BaseModel):
    name :str = Field(description= "Name of the Person")
    age : int = Field(gt =18,description="Age of the person")
    city : str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=person)

template =  PromptTemplate(
    template= "Gerate the name, age and city of a fictional {place} parson \n  {format_instruction}",
    input_variables=['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
    
)


chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)