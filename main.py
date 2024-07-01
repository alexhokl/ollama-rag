from pathlib import Path
import typer
from typing_extensions import Annotated
from ollama import get_grader
from ollama import get_model
from vectordb import retrieve_documents_from_directory
from vectordb import create_text_splitter
from vectordb import create_database
from vectordb import load_database

app = typer.Typer()

@app.command("load")
def load_documents_to_create_vector_db(
    documents_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    database_path: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ],
    database_name: Annotated[str, typer.Option()],
    file_pattern: str = "**/*.md",
    splitter_chunk_size: int = 1500,
    splitter_chunk_overlap: int = 300,
    ):

    print(f"About to load documents from [{documents_path}] with pattern [{file_pattern}] to create a vector database [{database_name}] at [{database_path}]")

    documents = retrieve_documents_from_directory(documents_path, file_pattern)
    print(f"Loaded {len(documents)} documents")
    splitter = create_text_splitter(chunk_size=splitter_chunk_size, chunk_overlap=splitter_chunk_overlap)
    _ = create_database(
        text_splitter=splitter,
        documents=documents,
        embedding_model="nomic-embed-text",
        database_name=database_name,
        database_path=database_path,
    )
    print("Vector database created")

@app.command("ask")
def ask(
    database_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    database_name: Annotated[str, typer.Option()],
    model_name: Annotated[str, typer.Option()],
    question: Annotated[str, typer.Option()],
    ):

    print(f"About to load vector database from [{database_path}] with name [{database_name}]")
    database = load_database(database_path, database_name)
    retriever = database.as_retriever()
    # print(f"Loaded vector database")

    grader = get_grader(model_name)
    # print(f"Loaded grader")

    llm = get_model(model_name, retriever)
    # print(f"Loaded model")

    documents = retriever.invoke(question)
    print(f"Loaded {len(documents)} documents from vector database for question [{question}]")

    filtered_docs = []
    for d in documents:
        score = grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            continue

    print(f"Number of documents to be used to answer the question: {len(filtered_docs)}")

    generated = llm.invoke({"context": filtered_docs, "question": question})
    print(f"Question: {question}")
    print(f"Answer: {generated.content}")

if __name__ == "__main__":
    app()
