from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TextSplitter

def retrieve_documents_from_directory(documents_path: Path, file_pattern: str) -> List[Document]:
    loader = DirectoryLoader(str(documents_path), glob=file_pattern, loader_cls=UnstructuredMarkdownLoader)
    return loader.load()

def create_text_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

def create_database(text_splitter: TextSplitter, documents: List[Document], embedding_model: str, database_name: str, database_path: Path) -> Chroma:
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=embedding_model, show_progress=False)
    vector_store = Chroma.from_documents(
        collection_name=database_name,
        documents=texts,
        embedding=embeddings,
        persist_directory=str(database_path),
    )
    return vector_store

def load_database(database_path: Path, database_name: str) -> Chroma:
    return Chroma(
        collection_name=database_name,
        persist_directory=str(database_path),
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )

def create_retirever(database: Chroma, search_type: str, result_count:int) -> VectorStoreRetriever:
    retriever = database.as_retriever(
        search_type=search_type,
        search_kwargs= {"k": result_count},
    )
    return retriever
