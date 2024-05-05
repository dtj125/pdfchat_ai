import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from transformers import InstructEmbedding
#from langchain.embeddings import HuggingFaceInstructEmbeddings
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

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
    embeddings = OpenAIEmbeddings()

    from InstructorEmbedding import INSTRUCTOR
    model = INSTRUCTOR('hkunlp/instructor-xl')
    sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    instruction = "Represent the Science title:"
    embeddings = model.encode([[instruction,sentence]])
    print(embeddings)


    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #model_name = "distilbert-base-uncased"  # Example model, you can replace it with any supported model
    #embeddings = InstructEmbedding.from_pretrained(model_name)
    vectorstore = FAISS.from_texts(texts=text_chucks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_icon=":books:", page_title="Chat with pdfs")

    st.header("CHAT WITH PDFS")
    st.text_input("ASk a question about your doc")

    #Side bar to upload docs
    with st.sidebar:
        st.subheader("Your docs")
        # store multiple files in variable
        pdf_docs = st.file_uploader("Upload PDF Here", accept_multiple_files=True)
        
        #Process button
        if st.button("process"):
            with st.spinner("Proccessing"): # spinning wheel on website to tell user its loading/processing
                # get pdf text -
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text) # write is like a print to see if the output is what you expect
                # get the text chunks
                text_chucks = get_text_chucks(raw_text)
                #st.write(text_chucks)
                # create a vector store - create embeddings for embedding model
                vectorstore = get_vectorstore(text_chucks)
                
if __name__ == '__main__':
    main()