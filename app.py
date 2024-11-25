#%%
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import ollama
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader



# current_dir = os.getcwd()
# persistent_db_path = os.path.join(current_dir, "dbtest")
#
# embedding_function = ollama.embeddings(model='nomic-embed-text')
#
# # Initialize the Chroma instance with the existing database
# db = Chroma(persist_directory=persistent_db_path,
#             collection_name="advanced_rag",
#             embedding_function=embedding_function)
#
#
# retriever = db.as_retriever()

#%%
file_path = os.getcwd() + "\\data\\data.txt"
loader = TextLoader(file_path, encoding="utf-8")
docs_word = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["##","\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
#%%
docs_split = text_splitter.split_documents(docs_word)
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
#%%
current_dir = os.path.dirname(os.getcwd() + "\\rag1")
persistent_db_path = os.path.join(current_dir, "dbtest567576")
db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)
#%%
db.add_documents(docs_split)
retriever = db.as_retriever()

#%%
def rag(user_query:str, language = "Polish", n_results=3):
    retriever = db.as_retriever()
    docs = retriever.invoke(user_query)[:n_results]
    print(docs)
    docs_content = [doc.page_content for doc in docs]
    joined_information = ';'.join([f'{doc.page_content}' for doc in docs])
    print(f"joined information: {joined_information}")
    messages = [
        ("system",
         "Jesteś zaawansowanym asystentem specjalizującym się w tworzeniu szczegółowych i realistycznych scenariuszy szkoleniowych dla wojska."
         "Będziesz otrzymywać od użytkownika konkretne pytania i odpowiednie informacje kontekstowe."
         "Na podstawie dostarczonych informacji:"
         "- Twórz szczegółowe i realistyczne scenariusze do szkolenia wojskowego."
         "„- Identyfikuj potencjalne zagrożenia, z jakimi mogą się zetknąć żołnierze podczas szkolenia i w trakcie rzeczywistych operacji."
         "- Wskazuj możliwe niespodziewane zmienne lub ryzyka, które mogą wystąpić w trakcie operacji."
         "- Przedstawiaj jasne, szczegółowe i praktyczne rekomendacje dotyczące tworzenia scenariuszy w celu minimalizacji ryzyka i poprawy przygotowania."
         "- Zwracaj uwagę na wszelkie luki w dostarczonych informacjach, które mogą wpływać na bezpieczeństwo lub dokładność scenariuszy."
         "Mów 'Nie wiem', jeśli nie jesteś pewien odpowiedzi na podstawie dostarczonych informacji."
         "Zawsze udzielaj wyczerpującej i precyzyjnej odpowiedzi w określonym języku."
         ),
        ("user", f"Odpowiedz na pytanie które zadał użytkownik:\n {user_query}. \n Na podstawie podanych informacji poniżej odpowiedz: \n {joined_information}. \n Language: {language}")
    ]
    print(f"messages: {messages}")
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatOllama(model="llama2:13b")
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({})
    # return also the complete prompt
    return docs_content, res, prompt.invoke({"query": user_query, "joined_information": joined_information, "language": language})
#%%
raw_docs, rag_response, prompt = rag("Na co zwracać uwagę podczas szturmu na okopy", language="Polish", n_results=3)
print(rag_response)
# raw_docs, rag_response, prompt = rag(query="Ist eine bestimmte Diät während der Radiotherapie erforderlich?", language="German", n_results=3)

# raw_docs, rag_response, prompt = rag(
#     "Żołnierze potrzebują szkolenia do obrony w mieście, "
#     "jesteś w stanie pomóc?",
#     language="Polish", n_results=5)
#
# print(rag_response)

#%%
st.header("Expert on Military Scenarios")

# text input field
user_query = st.text_input(label="User Query", help="Raise your questions You can ask in any language.", placeholder="What do you want to know?")


raw_docs = ["", "", "", "", ""]
rag_response = ""
prompt = None

language = st.selectbox("Output Language", ["English", "Polish"])


if st.button("Ask"):
    raw_docs, rag_response, prompt = rag(user_query, language="Polish", n_results=5)
    print(f"raw docs: {raw_docs}")
    print(f"rag response: {rag_response}")

# st.header("Retrieval")
# # st.markdown(f"**Raw Response 0:** {raw_docs[0]}")
# # st.markdown(f"**Raw Response 1:** {raw_docs[1]}")
# # st.markdown(f"**Raw Response 2:** {raw_docs[2]}")
# st.header("Augmentation")
# if prompt:
#     st.markdown(prompt.messages[1].content)

st.header("Generation")
st.write(rag_response)


