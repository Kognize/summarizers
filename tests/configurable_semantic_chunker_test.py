import unittest

from parameterized import parameterized
from langchain_openai import OpenAIEmbeddings

from summarizer.configurable_semantic_chunker import ConfigurableSemanticChunker


class MyTestCase(unittest.TestCase):
    @parameterized.expand([
        (85, 26),
        (90, 18),
        (95, 10)
    ])
    def test_summarizer(self, breakpoint_percentile_threshold: int, expected_number_of_chunks: int):
        # given
        text_splitter = ConfigurableSemanticChunker(OpenAIEmbeddings(),
                                                    breakpoint_percentile_threshold=breakpoint_percentile_threshold)
        with open("tests/resources/economy_of_singapore.txt") as philosophy_text:
            original_text = philosophy_text.read()

        # when
        chunks = text_splitter.create_documents([original_text])

        # then
        self.assertEqual(len(chunks), expected_number_of_chunks, "Split text should have expected number of chunks")
        print(f"percentile threshold: {breakpoint_percentile_threshold}, number of chunks: {len(chunks)}")
        lengths = [len(chunk.page_content) for chunk in chunks]
        lengths_str = ", ".join(str(length) for length in lengths)
        print(f"chunk lengths: {lengths_str}")


if __name__ == '__main__':
    unittest.main()
