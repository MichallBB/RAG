#%% packages
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
# from onnxruntime.transformers.shape_infer_helper import file_path
import ollama
load_dotenv(find_dotenv(usecwd=True))
import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
# %%

file_path = os.path.join(os.getcwd(), "data", "podreczniki_walki_eng.txt")

#%%
loader = TextLoader(file_path=file_path, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks_recursive = splitter.split_documents(docs)
len(chunks_recursive)

#%%
splitter = SemanticChunker(embeddings=OllamaEmbeddings(model='mxbai-embed-large'),
                           number_of_chunks=20)

chunks_semantic = splitter.split_documents(docs)

#%%
def create_chunks_with_context(document: Document):
    model = ChatOllama(model="llama2:13b")
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are part of a retrieval augmented generation pipeline. You are given a complete documents, and a chunk of the documents.
         Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval
          of the chunk. Answer only with the succinct context and nothing else. Don't repeat the chunk content.
         """),
        ("user", "<document>{document}</document><chunk>{chunk}</chunk>"),
    ])
    chain = prompt | model | StrOutputParser()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitter = SemanticChunker(embeddings=OllamaEmbeddings(model='mxbai-embed-large'),
                           number_of_chunks=20)
    chunks = splitter.split_documents([document])
    context_chunks = []
    for i, chunk in enumerate(chunks):
        print(i)
        response = chain.invoke({"document": document.page_content, "chunk": chunk})
        context_chunks.append(";".join([response, chunk.page_content]))

    return context_chunks

#%%
context_chunks = create_chunks_with_context(docs[0])
#%%
print(context_chunks[0])

#%%
len(context_chunks)

#%%
current_dir = os.getcwd()
persistent_db_path = os.path.join(current_dir, "dbsemantic")

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(persist_directory=persistent_db_path,
            collection_name="advanced_rag",
            embedding_function=embedding_function)



db.add_texts(context_chunks)

#%%
retriever = db.as_retriever()

# %%              What is Landing-attack operations
retriever.invoke("What is Offensive relief?")
#%%
db.get()

