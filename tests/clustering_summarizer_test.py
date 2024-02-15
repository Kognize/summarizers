import glob
import os
import unittest

from summarizer.clustering_summarizer import ClusteringSummarizer
from tests.utils.cosine_similarity import compare_semantic_similarity


def read_text_files(folder_path):
    # List to hold file contents
    file_contents = []
    # Use glob to find all text files in the folder
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Append file content to list
            file_contents.append(content)
    return file_contents


class MyTestCase(unittest.TestCase):
    def test_clustering_summarizer(self):
        # given
        with open("tests/resources/dracula/bram_strokers_dracula.txt") as dracula_full_book:
            original_text = dracula_full_book.read()

        summarizer = ClusteringSummarizer()

        # when
        result = summarizer.summarize(original_text)

        # then
        reference_summaries = read_text_files("tests/resources/dracula/summaries/")
        similarities = []
        for reference_summary in reference_summaries:
            similarities.append(compare_semantic_similarity(reference_summary, result))
        print(f"similarities: {similarities}")


if __name__ == '__main__':
    unittest.main()
