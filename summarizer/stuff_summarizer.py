from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.schema.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def summarize(text: str) -> str:
    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    document = Document(page_content=text, metadata={})

    return stuff_chain.run([document])
