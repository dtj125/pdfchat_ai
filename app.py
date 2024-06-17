import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# a class to create text embeddings using HuggingFace templates
from langchain.embeddings import HuggingFaceEmbeddings
#from transformers import InstructEmbedding
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS
#from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores.faiss import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.schema import Memory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
import os

# ✅ Set TOKENIZERS_PARALLELISM to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# take pdf docs and return a single string of all text from all pdfs
def get_pdf_text(pdf_docs):
    text = "" # variabe to store all the text from pdfs
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) # create pdf that has pages
        for page in pdf_reader.pages:
            text += page.extract_text() # Appended/Concat text to pages extraction text
    return text

def get_text_chucks(text):
    # text splitter by new line the take 1000 lines in one chuck the 
    # use chuck over lap in case you stop in the middle of a paragrapgh
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chucks = text_splitter.split_text(text) 
    return chucks

def get_vectorstore(text_chucks):
    #embeddings = OpenAIEmbeddings()    
    #model = INSTRUCTOR('hkunlp/instructor-xl')
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})
    #embeddings = INSTRUCTOR('hkunlp/instructor-xl')

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #vectorstore = FAISS.from_documents(texts=text_chucks, embedding=embeddings)
    vectorstore = FAISS.from_texts(texts=text_chucks, embedding=embeddings)

    return vectorstore

def get_conversation(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_icon=":books:", page_title="Chat with pdfs")
    st.write(css, unsafe_allow_html=True)

    # check if conversation already started
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("CHAT WITH PDFS")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    #Side bar to upload docs
    with st.sidebar:
        st.subheader("Your docs")
        # store multiple files in variable
        pdf_docs = st.file_uploader("Upload PDF Here", accept_multiple_files=True)
        
        #Process button
        if st.button("process"):
            with st.spinner("Proccessing"): # spinning wheel on website to tell user its loading/processing
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text) # write is like a print to see if the output is what you expect
                # get the text chunks
                text_chucks = get_text_chucks(raw_text)
                #st.write(text_chucks)
                # create a vector store - create embeddings for embedding model
                vectorstore = get_vectorstore(text_chucks)

                # Conversation chain creation - history of app
                # st.session_state will keep the the variables consistant so when it is updated variable stays the same
                #st.session_state.conversation = get_conversation(vectorstore)
                conversation = get_conversation(vectorstore)
                
if __name__ == '__main__':
    main()