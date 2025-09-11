import os
import argparse
import json
from porter import PorterStemmer
import re

# Path to stopwords file (relative to the corpus root)
STOPWORDS_FILE = "files/stopwords.txt"


# Load stopwords into a set for fast lookup
def load_stopwords(path):
    with open(path, 'r', encoding='latin-1') as f:
        return set(w.strip() for w in f)


# Tokenization function: lowercase and split text into alphanumeric tokens
def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Indexer:
    """
    BM25Indexer builds an inverted index from a collection of documents.
    It stores term frequencies per document, document lengths, and total document count.
    """

    def __init__(self, stopwords):
        self.stopwords = stopwords  # Set of stopwords to ignore
        self.stemmer = PorterStemmer()  # Porter stemmer for term normalization
        self.inverted = {}  # Inverted index: term -> {doc_id: frequency}
        self.doc_lengths = {}  # doc_id -> document length (number of valid terms)
        self.N = 0  # Total number of indexed documents

    def index_doc(self, doc_id, text):
        """
        Process a single document:
        - Tokenize
        - Remove stopwords
        - Stem terms
        - Update inverted index and document stats
        """
        # Preprocess: tokenize, remove stopwords, and stem
        terms = [self.stemmer.stem(t) for t in tokenize(text)
                 if t not in self.stopwords]

        # Record document length
        self.doc_lengths[doc_id] = len(terms)
        self.N += 1  # Increment total document count

        # Count term frequencies in the document
        counts = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1

        # Update inverted index
        for t, freq in counts.items():
            self.inverted.setdefault(t, {})[doc_id] = freq

    def save(self, path):
        """
        Save the index data to a JSON file, including:
        - Total document count
        - Document lengths
        - Inverted index
        """
        data = {
            'N': self.N,
            'doc_lengths': self.doc_lengths,
            'inverted': self.inverted
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for large corpus")
    parser.add_argument('-p', required=True, help='Path to large corpus directory')
    args = parser.parse_args()

    # Define input paths
    corpus_root = args.p
    stopwords_path = os.path.join(corpus_root, STOPWORDS_FILE)
    docs_dir = os.path.join(corpus_root, 'documents')

    # Load stopwords
    stopwords = load_stopwords(stopwords_path)

    # Create indexer instance
    indexer = BM25Indexer(stopwords)

    # Traverse all subdirectories and files in 'documents'
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            file_path = os.path.join(root, fname)

            # Skip non-regular files (just in case)
            if not os.path.isfile(file_path):
                continue

            # Read document text
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

            # Use filename (without extension) as document ID
            doc_id = os.path.splitext(fname)[0]
            indexer.index_doc(doc_id, text)

    # Save index to output file
    out_file = '22207232-large.index'
    indexer.save(out_file)
    print(f"Index saved to {out_file}")


if __name__ == '__main__':
    main()
