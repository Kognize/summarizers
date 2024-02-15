import unittest

from summarizer import stuff_summarizer
from tests.utils.cosine_similarity import compare_semantic_similarity


class MyTestCase(unittest.TestCase):
    def test_stuff_summarizer(self):
        # given
        with open("tests/resources/economy_of_singapore.txt") as economy_text:
            original_text = economy_text.read()

        # when
        result = stuff_summarizer.summarize(original_text)

        # then
        print(f"summarized text: \n{result}")
        similarity = compare_semantic_similarity(original_text, result)
        print(f"similarity: {similarity}")
        print(f"length of original text: {len(original_text)}")
        print(f"length of summarized text: {len(result)}")
        self.assertTrue(similarity > 0.7, "Similarity should be high.")
        self.assertTrue(len(result) < len(original_text) * 0.25,
                        "Summarized text should be shorter than 25% of original.")


if __name__ == '__main__':
    unittest.main()
