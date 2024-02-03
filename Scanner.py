import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import (CSVLoader, PyMuPDFLoader, TextLoader,
                                        UnstructuredEmailLoader,
                                        UnstructuredEPubLoader,
                                        UnstructuredHTMLLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredODTLoader,
                                        UnstructuredPowerPointLoader,
                                        UnstructuredWordDocumentLoader)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.docstore.document import Document
import pandas as pd
from io import StringIO
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import os
from multiprocessing import Pool
from typing import List
from tqdm import tqdm
from vector_database import CHROMA_SETTINGS
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()


#  Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY")
# directory where source documents to be ingested are located
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))
model_path = os.environ.get("MODEL_PATH")
chunk_size = 1000
chunk_overlap = 100

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = os.path.abspath(model_path),
        model_type="llama",
        callbacks=[StreamingStdOutCallbackHandler()],
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm



# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


st.title("Chat with CSV using Llama2 🦙🦜")

import chromadb


embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
ef = embedding_functions.DefaultEmbeddingFunction()


db = chromadb.PersistentClient(path = persist_directory)
collection = db.get_or_create_collection(name="my_collection", embedding_function=ef)


with st.sidebar:
    st.title('LLM chat App')
    add_vertical_space(5)
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, type=["pdf", "txt", "csv"])
    id = 1
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write(uploaded_file.name)      
        # st.write(bytes_data)  # Vous pouvez choisir de commenter cette ligne si vous ne souhaitez pas afficher le contenu du fichier

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                st.success("File uploaded successfully!")  # Ajout d'un message de succès si le fichier est téléchargé
            collection.add(documents=[tmp_file_path],ids=[str(id)])
            id += 1
            

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to say to your documents?"):
    # Display your message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add your message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # query ChromaDB based on your prompt, taking the top 5 most relevant result. These results are ordered by similarity.
    q = collection.query(
        query_texts=[prompt],
        n_results=5,
    )
    results = q["documents"][0]

    print(results)

    prompts = []
    for r in results:
        # construct prompts based on the retrieved text chunks in results 
        prompt = "Please extract the following: " + prompt + "  solely based on the text below. Use an unbiased and journalistic tone. If you're unsure of the answer, say you cannot find the answer. \n\n" + r

        prompts.append(prompt)
    prompts.reverse()

    llm = load_llm()
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    query = input("\nEnter a question: ")
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]
    print("\n\n> Question:")
    print(query)
    print(answer)