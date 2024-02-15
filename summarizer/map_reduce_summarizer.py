import langchain_openai
from langchain.chains import LLMChain, StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings

from summarizer.configurable_semantic_chunker import ConfigurableSemanticChunker


def summarize(text: str, breakpoint_percentile_threshhold=80) -> str:
    semantic_chunker = ConfigurableSemanticChunker(OpenAIEmbeddings(),
                                                   breakpoint_percentile_threshold=breakpoint_percentile_threshhold)
    documents = semantic_chunker.create_documents([text])
    print(f"Split the original document into {len(documents)} chunks.")

    llm = langchain_openai.chat_models.ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    return map_reduce_chain.run(documents)
