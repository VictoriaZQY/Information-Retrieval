import os
import argparse
import json
from porter import PorterStemmer
import re

# Path to the stopwords file relative to the corpus root
STOPWORDS_FILE = "files/stopwords.txt"

# Load stopwords into a set for efficient lookup
def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(w.strip() for w in f)

# Tokenization: split text into lowercase word tokens using regex
def tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    return tokens

class BM25Indexer:
    """
    BM25Indexer builds an inverted index mapping each term to document frequencies,
    tracks document lengths and the total number of documents.
    """
    def __init__(self, stopwords):
        # Set of stopwords to exclude during indexing
        self.stopwords = stopwords
        # PorterStemmer instance for term normalization
        self.stemmer = PorterStemmer()
        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted = {}
        # Document lengths: doc_id -> number of terms
        self.doc_lengths = {}
        # Total number of documents
        self.N = 0

    def index_doc(self, doc_id, text):
        """
        Tokenize, remove stopwords, stem terms, and update inverted index.
        """
        # Generate normalized terms
        terms = [self.stemmer.stem(t)
                 for t in tokenize(text)
                 if t not in self.stopwords]
        # Record the document length (number of terms)
        self.doc_lengths[doc_id] = len(terms)
        self.N += 1

        # Count term frequencies within this document
        counts = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1

        # Merge document term counts into global inverted index
        for t, freq in counts.items():
            self.inverted.setdefault(t, {})[doc_id] = freq

    def save(self, path):
        """
        Save index data (N, doc_lengths, inverted index) to a JSON file.
        """
        data = {
            'N': self.N,
            'doc_lengths': self.doc_lengths,
            'inverted': self.inverted
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def main():
    # Parse command-line arguments for corpus path
    parser = argparse.ArgumentParser(description="Build BM25 index for small corpus")
    parser.add_argument('-p', '--path', required=True, help='Path to small corpus directory')
    args = parser.parse_args()

    corpus_root = args.path
    stopwords_path = os.path.join(corpus_root, STOPWORDS_FILE)
    docs_dir = os.path.join(corpus_root, 'documents')

    # Load stopwords and initialize the indexer
    stopwords = load_stopwords(stopwords_path)
    indexer = BM25Indexer(stopwords)

    # Iterate over all document files and index each one
    for fname in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, fname)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        doc_id = os.path.splitext(fname)[0]
        indexer.index_doc(doc_id, text)

    # Save the index to the current working directory
    out_file = '22207232-small.index'
    indexer.save(out_file)
    print(f"Index saved to {out_file}")

if __name__ == '__main__':
    main()