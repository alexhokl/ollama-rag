from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema.runnable import RunnablePassthrough
from template import get_grader_template, get_model_template
from template import get_model_type

def get_grader(model_name: str):
    model_type = get_model_type(model_name)
    prompt_template = get_grader_template(model_type)
    grader_llm = ChatOllama(model=model_name, format="json", temperature=0)
    retriever = prompt_template | grader_llm | JsonOutputParser()
    return retriever

def get_model(model_name: str, retriever: VectorStoreRetriever, keep_alive: str = "3h", temperature: float = 0.0):
    model_type = get_model_type(model_name)
    prompt_template = get_model_template(model_type)
    llm = ChatOllama(
        model=model_name,
        keep_alive=keep_alive,
        temperature=temperature,
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )
    return rag_chain
