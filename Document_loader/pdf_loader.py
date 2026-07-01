import os
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('Difference between DBMS and RDBMS.pdf')
docs = loader.load()

print(docs[1].page_content)
# print(len(docs))
# print(docs[3].metadata)