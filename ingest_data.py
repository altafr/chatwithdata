from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

import PyPDF2

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        num_pages = pdf_reader.getNumPages()
        text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.getPage(page)
            page_text = page_obj.extractText()
            text += page_text
    return text

# loader = UnstructuredFileLoader("state_of_the_union.txt")
def embed_doc():
    #check data folder is not empty
    if len(os.listdir("data")) > 0:
        loader = DirectoryLoader('data', glob="**/*.txt")
        raw_documents = loader.load()
        print(len(raw_documents))
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
        )
        print("111")
        documents = text_splitter.split_documents(raw_documents)


        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        print("222")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("333")


        # Save vectorstore
        # check if vectorstore.pkl exists
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)


# check if vectorstore.pkl exists
if  os.path.exists("vectorstore.pkl"):
    with open("vectorstore.pkl", 'rb') as f:
        docsearch = pickle.load(f)

# query = input("Enter your query: ")
# docs = docsearch.similarity_search(query)
# print(docs[0])
