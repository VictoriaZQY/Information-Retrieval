import os
import argparse
import json
import math
import re
from porter import PorterStemmer

class BM25Query:
    """
    BM25Query loads a precomputed index and computes BM25 scores
    for an input query against all indexed documents.
    """
    def __init__(self, index_path):
        # Load JSON index file
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.N = data['N']  # Total number of documents
        self.doc_lengths = data['doc_lengths']  # Document lengths
        self.inverted = data['inverted']  # Inverted index: term -> {doc_id: frequency}
        self.avg_len = sum(self.doc_lengths.values()) / self.N  # Average document length
        self.k1 = 1  # BM25 parameter k1
        self.b = 0.75  # BM25 parameter b
        self.stemmer = PorterStemmer()  # Porter stemmer for query preprocessing

    def score(self, query):
        """
        Compute BM25 score for each document given the input query.
        Returns a sorted list of (doc_id, score) tuples in descending order.
        """
        # Tokenize and stem the query terms
        q_terms = [self.stemmer.stem(t) for t in re.findall(r"\w+", query.lower())]
        scores = {}

        # Accumulate BM25 score contributions per document
        for t in q_terms:
            postings = self.inverted.get(t, {})
            df = len(postings)
            idf = math.log((self.N - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings.items():
                dl = self.doc_lengths[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                term_score = idf * (freq * (self.k1 + 1) / denom)
                scores[doc_id] = scores.get(doc_id, 0) + term_score

        # Return documents sorted by score (descending)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Query large corpus using BM25 index")
    parser.add_argument('-m', required=True, choices=['interactive', 'automatic'],
                        help='Mode: interactive or automatic')
    parser.add_argument('-p', required=True, help='Path to large corpus directory')
    args = parser.parse_args()

    # Path to index file (must be prebuilt)
    index_file = '22207232-large.index'
    bm25 = BM25Query(index_file)

    if args.m == 'interactive':
        # Interactive mode: user types query and gets Top 10 results
        while True:
            q = input('Enter query: ')
            if q.upper() == 'QUIT':
                break
            results = bm25.score(q)[:10]  # Show only Top 10 results
            for rank, (doc, sc) in enumerate(results, start=1):
                print(rank, doc, f"{sc:.6f}")
    else:
        # Automatic mode: load all queries and generate result file
        qfile = os.path.join(args.p, 'files', 'queries.txt')
        out_path = '22207232-large.results'
        with open(qfile, 'r', encoding='utf-8') as fin, \
             open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                qid, *terms = line.strip().split()
                q = ' '.join(terms)
                ranked = bm25.score(q)[:1000]  # Top 1000 for evaluation
                for rank, (doc, sc) in enumerate(ranked, start=1):
                    fout.write(f"{qid} {doc} {rank} {sc:.6f}\n")
        print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
