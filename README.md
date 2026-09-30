# Information-Retrieval
Although this project was finished on June 2nd, 2025, it was uploaded later because the large corpus has too many files.

Evaluate information retrieval methods, including Precision, Recall, R-Precision, P@15, NDCG@15, MAP, and bpref, on both small and large corpora. 

Here we have an introduction on how to run this project:
https://github.com/VictoriaZQY/Information-Retrieval/blob/main/single/COMP3009J-corpus-large/README.md

The main objective of the assignment is to create a basic Information Retrieval system that can perform preprocessing, indexing, retrieval (using BM25) and evaluation.
The small corpus is intended to show the correctness of your code. The large corpus is intended to show the efficiency. Efficiency is only important if the code is firstly correct.
Both corpora are in the same format, except for the relevance judgments. For the small corpus, all documents not included in the relevance judgments have been judged non-relevant. For the large corpus, documents not included in the relevance judgments have not been judged.

## index_small_corpus.py
This program is intended to read the small corpus, process its contents and create an index.

It must be possible to pass the path to the (unzipped) small corpus to this program as a command-line argument named “-p”3:
./index_small_corpus.py -p /path/to/comp3009j-corpus-small

This program must perform the following tasks:
1. Extract the documents contained in the corpus provided. You must divide the documents into terms in an appropriate way (these are contained in the ``documents’’ directory of the corpus. The strategy must be documented in your source code comments.
2. Perform stopword removal. A list of stopwords to use can be loaded from the stopwords.txt file that is provided in the ``files’’ directory of the corpus.
3. Perform stemming. For this task, you may use the porter.py code in the ``files’’ directory.
4. Create an appropriate index so that IR using the BM25 method may be performed. Here, an index is any data structure that is suitable for performing retrieval later.

This will require you to calculate the appropriate weights and do as much pre-calculation as you can. This should be stored in a single external file in some human-readable4 format. Do not use database systems (e.g. MySQL, SQL Server, SQLite, etc.) for this.

The output of this program should be a single index file, stored in the current working directory.

## query_small_corpus.py
This program allows a user to submit queries to retrieve from the small corpus, or to run the standard corpus queries so that the system can be evaluated. The BM25 model must be used for retrieval.
Every time this program runs, it should first load the index into memory (named “21888888-small.index” in the current working directory, replacing “21888888” with your UCD student number), so that querying can be as fast as possible.
This program should offer two modes, depending on a command-line argument named “-m”. These are as follows:
1. Interactive mode: 
In this mode, a user can manually type in queries and see the first 15 results in their command line, sorted beginning with the highest similarity score. The output should have three columns: the rank, the document’s ID, and the similarity score. A sample run of the program is contained later in this document. The user should continue to be prompted to enter further queries until they type “QUIT”.
Example output is given below. Interactive mode is activated by running the program in the following way: ./query_small_corpus.py -m interactive -p /path/to/comp3009j-corpus-small
2. Automatic mode: 
In this mode, the standard queries should be read from the "queries.txt" file (in the "files" directory of the corpus). This file has a query on each line, beginning with its query ID. The results5 should be stored in a file named “218888880-small.results" in the current working directory (replacing “21888888” with your UCD student number), which should include four columns: query ID, document ID, rank and similarity score. A sample of the desired output can be found in the “sample_output.txt” file in the “files” directory in the corpus.
Automatic mode is activated by running the program in the following way: 
./query_small_corpus.py -m automatic -p /path/to/comp3009j-corpus-small

## evaluate_small_corpus.py
This program calculates suitable evaluation metrics, based on the output of the automatic mode of query_small_corpus.py (stored in “218888880-small.results" in the current working directory (replacing “21888888” with your UCD student number). The program should calculate the following metrics, based on the relevance judgments contained in the "qrels.txt" file in the "files" directory of the corpus):
- Precision
- Recall
- R-Precision
- P@15
- NDCG@15
- MAP

The program should be run in the following way:
./evaluate_small_corpus.py -p /path/to/comp3009j-corpus-small

## index_large_corpus.py
This program should perform the same tasks as index_small_corpus.py.

## query_large_corpus.py
This program should perform the same tasks as query_small_corpus.py.

## evaluate_large_corpus.py
In addition to the evaluation metrics calculated by evaluate_small_corpus.py, this program should also calculate bpref (since the large corpus has incomplete relevance judgments).
Otherwise, this program should perform the same tasks as evaluate_small_corpus.py.
