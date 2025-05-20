from bert_score import score


def compute_bertscore(results, ground_truth):
    """Compute BERTScore using full pairwise comparison between generated and actual recommendation lists."""

    if len(results) == 0 or len(ground_truth) == 0:
        return {
            "Precision": "#NULL",
            "Recall": "#NULL",
            "F1-score": "#NULL"
        }

    try:
        all_hypotheses = []
        all_references = []

        for res in results:
            for ref in ground_truth:
                all_hypotheses.append(res)
                all_references.append(ref)

        P_all, R_all, F1_all = score(all_hypotheses, all_references, lang="en", rescale_with_baseline=True, model_type="distilbert-base-uncased")

        n = len(results)
        m = len(ground_truth)
        P_matrix = P_all.view(n, m)
        R_matrix = R_all.view(n, m)

        # precision: for each generated item, find best match in ground truth
        P_vals = P_matrix.max(dim=1).values

        # recall: for each ground truth item, find best match in generated
        R_vals = R_matrix.max(dim=0).values

        precision = P_vals.mean().item()
        recall = R_vals.mean().item()

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }
    except AssertionError:
        return {
            "Precision": "#NULL",
            "Recall": "#NULL",
            "F1-score": "#NULL"
        }
