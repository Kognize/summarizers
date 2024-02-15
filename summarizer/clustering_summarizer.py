from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ClusteringSummarizer:
    def __init__(self):
        self.recursive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            length_function=len,
            is_separator_regex=False,
        )

    def summarize(self, text: str) -> str:
        chunks = self.recursive_text_splitter.create_documents([text])
        print(f"original text split into {str(len(chunks))} chunks")

        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents([x.page_content for x in chunks])

        return "hehe"
