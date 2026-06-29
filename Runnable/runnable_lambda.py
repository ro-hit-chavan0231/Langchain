from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


model = ChatOllama(model = "tinyllama")

parser = StrOutputParser()

def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template='write 3-5 line of joke about {topic}',
    input_variables=['topic']
)

joke_gen_chian = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chian, parallel_chain)

print(final_chain.invoke({'topic':'AI'}))