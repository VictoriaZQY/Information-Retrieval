import argparse
import math
import os


def load_qrels(qrels_file):
    """
    Load relevance judgments from qrels.txt
    Params:
        qrels_file (str): path to the qrels.txt file
    Returns:
        dict: { query_id: { doc_id: relevance_score, ... }, ... }
    """
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            # Initialize nested dict if first time seeing this query
            if qid not in qrels:
                qrels[qid] = {}
            # Store the relevance score (0 = non-relevant, >0 = relevant)
            qrels[qid][docid] = rel
    return qrels


def load_results(results_file):
    """
    Load system retrieval results from .results file
    Params:
        results_file (str): path to the results file
    Returns:
        dict: { query_id: [doc1, doc2, ...], ... }
    """
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, docid = parts[0], parts[1]
            # Append each retrieved document in ranked order
            results.setdefault(qid, []).append(docid)
    return results


def precision_at_n(retrieved, relevant_set, n):
    """
    Compute Precision@N: fraction of top-N retrieved docs that are relevant.
    Denominator adjusts if fewer than N docs returned.
    """
    # consider top-N retrieved documents (may be fewer)
    top_n = retrieved[:n]
    if not top_n:
        return 0.0
    # count relevant among them
    num_rel = sum(1 for d in top_n if d in relevant_set)
    return num_rel / len(top_n)


def r_precision(retrieved, relevant_set):
    """
    Compute R-Precision: precision at R, where R = number of relevant docs for this query.
    """
    R = len(relevant_set)
    if R == 0:
        return 0.0
    # precision for top-R documents
    return precision_at_n(retrieved, relevant_set, R)


def average_precision(retrieved, relevant_set):
    """
    Compute Average Precision (AP) for one query.
    Sum precision at each rank where a relevant document is found,
    then divide by total number of relevant docs.
    """
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            # precision up to this rank: hits / idx
            sum_prec += hits / idx
    # normalize by total relevant count
    return sum_prec / len(relevant_set)


def ndcg_at_n(retrieved, rel_dict, n):
    """
    Compute Normalized Discounted Cumulative Gain @N (NDCG@N).
    DCG: sum of graded relevance / log2(rank)
    IDCG: ideal DCG by sorting relevance scores in descending.
    Then NDCG = DCG@N / IDCG@N
    """
    # DCG calculation
    dcg = 0.0
    for i, doc in enumerate(retrieved[:n], start=1):
        rel = rel_dict.get(doc, 0)
        # first rank (i=1) uses rel, subsequent use discounted
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i)
    # IDCG: sort ideal relevance scores descending
    ideal_rels = sorted(rel_dict.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:n], start=1):
        if i == 1:
            idcg += rel
        else:
            idcg += rel / math.log2(i)
    # if no ideal gain, return 0
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_recall(results, qrels):
    """
    Compute overall Precision and Recall averaged over all queries.
    Precision: proportion of retrieved docs that are relevant.
    Recall: proportion of relevant docs that are retrieved.
    """
    total_prec = 0.0
    total_rec = 0.0
    Q = len(results)
    for qid, retrieved in results.items():
        relevant_set = set(qrels.get(qid, {}).keys())
        if not retrieved:
            continue
        # count relevant retrieved
        num_rel = sum(1 for d in retrieved if d in relevant_set)
        # per-query precision and recall
        prec = num_rel / len(retrieved)
        rec = num_rel / len(relevant_set) if relevant_set else 0.0
        total_prec += prec
        total_rec += rec
    return total_prec / Q, total_rec / Q


def calculate_map(results, qrels):
    """
    Compute Mean Average Precision (MAP) over all queries.
    """
    total_ap = 0.0
    Q = len(results)
    for qid, retrieved in results.items():
        # use previously defined AP function
        ap = average_precision(retrieved, set(qrels.get(qid, {}).keys()))
        total_ap += ap
    return total_ap / Q


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate small corpus')
    parser.add_argument('-p', required=True,
                        help='Path to comp3009j-corpus-small directory')
    args = parser.parse_args()

    # Paths to qrels and results files
    qrels_file = os.path.join(args.p, 'files', 'qrels.txt')
    results_file = os.path.join(os.getcwd(), '22207232-small.results')

    # Load data
    qrels = load_qrels(qrels_file)
    results = load_results(results_file)

    # Compute metrics
    precision, recall = calculate_precision_recall(results, qrels)
    rprec = r_precision_for_all = sum(r_precision(res, set(qrels[qid].keys())) \
                                      for qid, res in results.items()) / len(results)
    # R-Precision averaged similarly
    # Note: direct loop since r_precision defined per query
    p_at_15 = sum(precision_at_n(res, set(qrels[qid].keys()), 15)
                  for qid, res in results.items()) / len(results)
    ndcg15 = sum(ndcg_at_n(res, qrels.get(qid, {}), 15)
                 for qid, res in results.items()) / len(results)
    map_score = calculate_map(results, qrels)

    # Print results in required order
    print('Evaluation results:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'R-Precision: {rprec:.4f}')  # average R-Precision
    print(f'P@15: {p_at_15:.4f}')
    print(f'NDCG@15: {ndcg15:.4f}')
    print(f'MAP: {map_score:.4f}')


if __name__ == '__main__':
    main()
