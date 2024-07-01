from enum import Enum
from langchain_core.prompts import PromptTemplate

class ModelType(str, Enum):
    llama3 = "llama3"
    gemma2 = "gemma2"

def get_model_type(model_name: str) -> ModelType:
    if model_name.find("llama3") >= 0:
        return ModelType.llama3
    if model_name.find("gemma2") >= 0:
        return ModelType.gemma2
    raise ValueError(f"Unsupported model: {model_name}")

def get_model_template(model_type: ModelType) -> PromptTemplate:
    match model_type:
        case ModelType.llama3:
            return PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant for question-answering tasks. \
                    Use the following pieces of retrieved context to answer the question. \
                    Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
                    If the context doen't contain the answer, just respond that you are unable to find an answer. \
                    Replace "the text" or "the context" or "the provided text" with "our documentation".
                    Explain the reasoning as well.<|eot_id|><|start_header_id|>user<|end_header_id|>

                    Context: {context}

                    Question: {question}

                    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|> """,
                input_variables=["question", "document"],
            )
        case ModelType.gemma2:
            return PromptTemplate(
                template="""<bos><start_of_turn>user\nYou are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
                    Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
                    If the context doen't contain the answer, just respond that you are unable to find an answer. \
                    Replace "the text" or "the context" or "the provided text" with "our documentation".

                    CONTEXT: {context}

                    QUESTION: {question}

                    <end_of_turn>
                    <start_of_turn>model\n
                    ANSWER:""",
                input_variables=["question", "document"],
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

def get_grader_template(model_type: ModelType) -> PromptTemplate:
    match model_type:
        case ModelType.llama3:
            return PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
                    of a retrieved document to a user question. If the document contains keywords related to the user question,
                    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    Here is the retrieved document: \n\n {document} \n\n
                    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question", "document"],
            )
        case ModelType.gemma2:
            return PromptTemplate(
                template="""<bos><start_of_turn>user\nYou are a grader assessing relevance
                    of a retrieved document to a user question. If the document contains keywords related to the user question,
                    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
                    Here is the retrieved document: \n\n {document} \n\n
                    Here is the user question: {question} \n <end_of_turn>
                    <start_of_turn>model\n""",
                input_variables=["question", "document"],
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


