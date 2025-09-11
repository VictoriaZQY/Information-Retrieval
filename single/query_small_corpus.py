import os
import argparse
import json
import math
import re
from porter import PorterStemmer

class BM25Query:
    """
    A class that performs BM25 scoring using a pre-built index.
    """
    def __init__(self, index_path):
        # Load the index file (contains N, doc_lengths, and inverted index)
        with open(index_path) as f:
            data = json.load(f)
        self.N = data['N']  # Total number of documents
        self.doc_lengths = data['doc_lengths']  # Dict: doc_id -> length
        self.inverted = data['inverted']  # Dict: term -> {doc_id: frequency}
        self.avg_len = sum(self.doc_lengths.values()) / self.N  # Average document length
        self.k1 = 1  # BM25 parameter
        self.b = 0.75  # BM25 parameter
        self.stemmer = PorterStemmer()

    def score(self, query):
        """
        Compute BM25 scores for a query string against the index.
        Returns a sorted list of (doc_id, score) tuples.
        """
        # Tokenize and stem query terms
        q_terms = [self.stemmer.stem(t) for t in re.findall(r"\w+", query.lower())]
        scores = {}

        for t in q_terms:
            postings = self.inverted.get(t, {})  # Retrieve doc -> freq dict
            df = len(postings)  # Document frequency
            idf = math.log((self.N - df + 0.5) / (df + 0.5))

            for doc_id, freq in postings.items():
                dl = self.doc_lengths[doc_id]  # Document length
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                sim = idf * (freq * (self.k1 + 1) / denom)
                scores[doc_id] = scores.get(doc_id, 0) + sim  # Accumulate score

        # Return sorted list by descending score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', required=True, choices=['interactive', 'automatic'],
                        help='Mode: interactive or automatic')
    parser.add_argument('-p', required=True, help='Path to corpus directory')
    args = parser.parse_args()

    # Load index from file
    index_file = '22207232-small.index'
    bm25 = BM25Query(index_file)

    if args.m == 'interactive':
        print("Enter queries (type 'QUIT' to exit)")
        while True:
            q = input('Enter query: ')
            if q.upper() == 'QUIT':
                break
            results = bm25.score(q)[:10]  # Show top 10 results
            for rank, (doc, sc) in enumerate(results, start=1):
                print(rank, doc, f"{sc:.6f}")
    else:
        # Automatic mode: read queries from file and output results
        qfile = os.path.join(args.p, 'files', 'queries.txt')
        out = open('22207232-small.results', 'w')
        for line in open(qfile):
            qid, *terms = line.strip().split()
            q = ' '.join(terms)
            res = bm25.score(q)[:1000]  # Top 1000 for TREC-style eval
            for rank, (doc, sc) in enumerate(res, start=1):
                out.write(f"{qid} {doc} {rank} {sc:.6f}\n")
        out.close()
        print("Results written to 22207232-small.results")

if __name__ == '__main__':
    main()
