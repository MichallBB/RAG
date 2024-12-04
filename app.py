#%%

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from sympy.physics.units import temperature

load_dotenv(find_dotenv(usecwd=True))


#%%
current_dir = os.getcwd()
persistent_db_path = os.path.join(current_dir, "dbtestnewv2")

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(persist_directory=persistent_db_path,
            embedding_function=embedding_function)

retriever = db.as_retriever()

#%%
def rag(user_query:str, language = "Polish", n_results=3):
    docs = retriever.invoke(user_query)[:n_results]
    #print(docs)
    docs_content = [doc.page_content for doc in docs]
    joined_information = ';'.join([f'{doc.page_content}' for doc in docs])
    print(f"joined information: {joined_information}")
    print(f"\n\n\n\n {user_query}")
    messages = [
        ("system",
         "You are a military assistant.Your users are asking questions about information contained in attached information."
         "You will be shown the user's question, and the relevant information. Answer the user's question using only this information."
         "Say 'I don't know' if you don't know the answer.Answer in specified language."
         "Always provide long, comprehensive and precise answer. Answer only if question is related to the given information."
         ),
        ("user", f"Answer for this question:\n {user_query}. \n Use the given information for answer: \n {joined_information}.")
    ]
#    print(f"messages: {messages}")
    prompt = ChatPromptTemplate.from_messages(messages)
    #model = ChatOllama(model="llama2:13b", temperature=0.4)
    model = ChatOllama(model="llama3.1:8b", temperature=0.5)
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({})
    # return also the complete prompt
    return docs_content, res, prompt.invoke({"query": user_query, "joined_information": joined_information, "language": language})


#%%
st.header("")

# text input field
user_query = st.text_input(label="User Query", help="Raise your questions You can ask in any language.", placeholder="What do you want to know?")


raw_docs = ["", "", "", "", ""]
rag_response = ""
prompt = None

language = st.selectbox("Output Language", ["English", "Polish"])


if st.button("Ask"):
    raw_docs, rag_response, prompt = rag(user_query, language="Polish", n_results=5)
#    print(f"raw docs: {raw_docs}")
#    print(f"rag response: {rag_response}")

# st.header("Retrieval")
# # st.markdown(f"**Raw Response 0:** {raw_docs[0]}")
# # st.markdown(f"**Raw Response 1:** {raw_docs[1]}")
# # st.markdown(f"**Raw Response 2:** {raw_docs[2]}")
st.header("Augmentation")
#if prompt:
#    st.markdown(prompt.messages[1].content)

st.header("Generation")
st.write(rag_response)

# %%              What is Landing-attack operations
#retriever.invoke("What is Offensive relief?")
