import argparse
import math
import os


def load_qrels(qrels_file):
    """
    Load relevance judgments from qrels.txt
    Params:
        qrels_file (str): path to the qrels.txt file
    Returns:
        dict: { query_id: { doc_id: relevance_score (>=0), ... }, ... }
    """
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            # Initialize nested dict if first time seeing this query
            qrels.setdefault(qid, {})
            # Store each judgment (0 = non-relevant, >0 = relevant grade)
            qrels[qid][docid] = rel
    return qrels


def load_results(results_file):
    """
    Load retrieval results from .results file
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
            results.setdefault(qid, []).append(docid)
    return results


def precision_at_n(retrieved, relevant_set, n):
    """
    Compute Precision@N: fraction of top-N retrieved docs that are relevant.
    If fewer than N docs returned, denominator = number returned.
    """
    top_n = retrieved[:n]
    if not top_n:
        return 0.0
    return sum(1 for d in top_n if d in relevant_set) / len(top_n)


def r_precision(retrieved, relevant_set):
    """
    Compute R-Precision: precision at R, where R = total number of relevant docs.
    """
    R = len(relevant_set)
    if R == 0:
        return 0.0
    return precision_at_n(retrieved, relevant_set, R)


def average_precision(retrieved, relevant_set):
    """
    Compute Average Precision (AP):
    Sum precision at each rank where a relevant doc is found,
    then divide by total number of relevant docs.
    """
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            sum_prec += hits / idx
    return sum_prec / len(relevant_set)


def ndcg_at_n(retrieved, rel_dict, n):
    """
    Compute NDCG@N using vector-based approach:
    - Build DCG vector incrementally: DCG[i] = cumulative gain up to rank i
    - Build IDCG vector from ideal sorted relevance grades
    - Pad IDCG if fewer ideal entries than N
    - Normalise: ndcg[i] = DCG[i] / IDCG[i] if IDCG[i] > 0
    Return value at rank N (index N-1).
    """
    # Build DCG sequence
    dcg = []
    for i, doc in enumerate(retrieved[:n]):
        gain = rel_dict.get(doc, 0)
        if i == 0:
            dcg.append(gain)
        else:
            dcg.append(dcg[i-1] + gain / math.log2(i+1))
    # Build ideal DCG sequence
    ideal_rels = sorted(rel_dict.values(), reverse=True)
    idcg = []
    for i, gain in enumerate(ideal_rels[:n]):
        if i == 0:
            idcg.append(gain)
        else:
            idcg.append(idcg[i-1] + gain / math.log2(i+1))
    # Extend IDCG to length n if needed
    while len(idcg) < n:
        idcg.append(idcg[-1] if idcg else 0.0)
    # Compute NDCG sequence and return at N
    ndcg = [dcg[i] / idcg[i] if idcg[i] > 0 else 0.0 for i in range(n)]
    return ndcg[n-1]


def bpref(retrieved, rel_dict, nonrel_set):
    """
    Compute Bpref for incomplete judgments:
    For each relevant doc, count non-relevant docs ranked higher,
    cap nonrel_before at R to avoid negative contributions,
    then average (1 - nonrel_before/R) over all R relevant docs.
    """
    R = len(rel_dict)
    if R == 0:
        return 0.0
    sum_b = 0.0
    nonrel_before = 0
    rel_count = 0
    for doc in retrieved:
        if doc in nonrel_set:
            nonrel_before = min(nonrel_before + 1, R)
        if doc in rel_dict:
            # contribution for this relevant doc
            sum_b += 1.0 - nonrel_before / R
            rel_count += 1
            if rel_count == R:
                break  # all relevant docs processed
    return sum_b / R


def calculate_precision_recall(results, qrels):
    """
    Compute overall Precision and Recall averaged over all queries.
    Precision: proportion of retrieved docs that are relevant.
    Recall: proportion of relevant docs that are retrieved.
    """
    total_p, total_r = 0.0, 0.0
    Q = len(results)
    for qid, retrieved in results.items():
        rel_docs = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        if not retrieved:
            continue
        # count relevant retrieved
        num_rel = sum(1 for d in retrieved if d in rel_docs)
        total_p += num_rel / len(retrieved)
        total_r += num_rel / len(rel_docs) if rel_docs else 0.0
    return total_p / Q, total_r / Q


def calculate_map(results, qrels):
    """
    Compute Mean Average Precision (MAP) over all queries.
    """
    total_ap = 0.0
    Q = len(results)
    for qid, retrieved in results.items():
        rel_docs = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        total_ap += average_precision(retrieved, rel_docs)
    return total_ap / Q


def calculate_bpref(results, qrels):
    """
    Compute average Bpref across queries for incomplete judgments.
    """
    total_bp = 0.0
    Q = len(results)
    for qid, retrieved in results.items():
        rel_dict = {d: rel for d, rel in qrels.get(qid, {}).items() if rel > 0}
        nonrel_set = {d for d, rel in qrels.get(qid, {}).items() if rel == 0}
        total_bp += bpref(retrieved, rel_dict, nonrel_set)
    return total_bp / Q


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate large corpus')
    parser.add_argument('-p', required=True, help='Path to comp3009j-corpus-large directory')
    args = parser.parse_args()

    # Paths to qrels and results files
    qrels_file = os.path.join(args.p, 'files', 'qrels.txt')
    results_file = os.path.join(os.getcwd(), '22207232-large.results')

    # Load data
    qrels = load_qrels(qrels_file)
    results = load_results(results_file)

    # Compute metrics
    precision, recall = calculate_precision_recall(results, qrels)
    rprec = sum(r_precision(results[qid], {d for d, rel in qrels[qid].items() if rel > 0}) for qid in results) / len(results)
    p15 = sum(precision_at_n(results[qid], {d for d, rel in qrels[qid].items() if rel > 0}, 15) for qid in results) / len(results)
    ndcg15 = sum(ndcg_at_n(results[qid], qrels[qid], 15) for qid in results) / len(results)
    map_score = calculate_map(results, qrels)
    bp_score = calculate_bpref(results, qrels)

    # Print results in required order
    print('Evaluation results:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'R-Precision: {rprec:.4f}')
    print(f'P@15: {p15:.4f}')
    print(f'NDCG@15: {ndcg15:.4f}')
    print(f'MAP: {map_score:.4f}')
    print(f'Bpref: {bp_score:.4f}')

if __name__ == '__main__':
    main()
