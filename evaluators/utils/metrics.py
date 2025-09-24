from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy

def calc_primevul_score(ground_truth: list, predictions: list) -> dict:
    """
    Calculate evaluation metrics for vulnerability detection that handles special prediction values.
    
    Args:
        ground_truth: List of ground truth labels (0 or 1)
        predictions: List of predicted labels that may contain -2 (incorrect), -1 (empty), 0 (benign), 1 (vulnerable)
    
    Returns:
        dict: Dictionary containing metrics including:
            - accuracy, precision, recall, f1
            - empty_predictions: count of -1 predictions
            - tp, tn, fp, fn: detailed counts for analysis
    """
    tp = tn = fp = fn = empty_cnt = 0
    
    for gt, pred in zip(ground_truth, predictions):
        if pred in [-1, -2]:  # Handle empty or incorrect predictions
            empty_cnt += 1
            if gt == 1:
                fn += 1
        else:  # Normal binary classification
            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 1 and pred == 0:
                fn += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 0 and pred == 0:
                tn += 1

    total = len(ground_truth)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    scores = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'empty_predictions': empty_cnt,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return scores

def calc_primevul_pair_score(ground_truth_list: list, pair_predictions: list) -> dict:
    """

    Args:
        datum1_ground_truth (int): _description_
        datum1_prediction (int): _description_
        datum2_ground_truth (int): _description_
        datum2_prediction (int): _description_

    Returns:
        Pair-wise Correct Prediction (P-C): The model correctly predicts the ground-truth labels for both elements of a pair.
        Pair-wise Vulnerable Prediction (P-V): The model incorrectly predicts both elements of the pair as vulnerable.
        Pair-wise Benign Prediction (P-B): The model incorrectly predicts both elements of the pair as benign.
        Pair-wise Reversed Prediction (P-R): The model incorrectly and inversely predicts the labels for the pair.
        
    """
    score = {
        "P-C": 0,
        "P-V": 0,
        "P-B": 0,
        "P-R": 0,
    }
    # print(pair_predictions)
    count = 1
    for datum_prediction, ground_truth in zip(pair_predictions, ground_truth_list):
        # print(count)
        # count += 1
        # print(datum_prediction)
        datum1_prediction, datum2_prediction = datum_prediction
        datum1_ground_truth, datum2_ground_truth = ground_truth
        # print(datum1_prediction, datum2_prediction, datum1_ground_truth, datum2_ground_truth)
        if datum1_ground_truth == datum1_prediction and datum2_ground_truth == datum2_prediction:
            score["P-C"] += 1
        if datum1_prediction == 1 and datum2_prediction == 1:
            score["P-V"] += 1
        if datum1_prediction == 0 and datum2_prediction == 0:
            score["P-B"] += 1
        if (datum1_ground_truth == 1 and datum1_prediction == 0 and datum2_ground_truth == 0 and datum2_prediction == 1) or \
            (datum1_ground_truth == 0 and datum1_prediction == 1 and datum2_ground_truth == 1 and datum2_prediction == 0):
            score["P-R"] += 1

    raw_score = copy.deepcopy(score)
    total = sum(score.values())
    for k, v in score.items():
        score[k] = v / total
    # print(raw_score)
    return raw_score, score
