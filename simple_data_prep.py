#%% packages
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
#%% data loading
file_path = os.getcwd() + "\\data\\podreczniki_walki_eng.txt"
loader = TextLoader(file_path, encoding="utf-8")
docs_word = loader.load()


# %% visual inspection
pprint(docs_word[0].page_content[:400])

#%% text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["##","\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
docs_split = text_splitter.split_documents(docs_word)

# %% find the number of chunks
len(docs_split)
# %%
embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
#%%
current_dir = os.path.dirname(os.getcwd() + "\\rag1")
persistent_db_path = os.path.join(current_dir, "dbtestnewv2")

db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)


# %%
db.add_documents(docs_split)
# %%
len(db.get()['ids'])


# %% set up a retriever
retriever = db.as_retriever()
# %%              What is Landing-attack operations
#retriever.invoke("What is Offensive relief?")
retriever.invoke("a")

# %%