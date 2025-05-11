from bert_score import score


def compute_bertscore(results, ground_truth):
    """Compute BERTScore between generated and actual recommendations."""

    min_len = min(len(results), len(ground_truth))
    if min_len == 0:
        return {
            "Precision": "#NULL",
            "Recall": "#NULL",
            "F1-score": "#NULL"
        }

    results = results[:min_len]
    ground_truth = ground_truth[:min_len]
    try:
        P, R, F1 = score(results, ground_truth, lang="en", rescale_with_baseline=True)  # using the default 'roberta-large'
        return {
            "Precision": P.mean().item(),
            "Recall": R.mean().item(),
            "F1-score": F1.mean().item()
        }
    except AssertionError:
        return {
            "Precision": "#NULL",
            "Recall": "#NULL",
            "F1-score": "#NULL"
        }
