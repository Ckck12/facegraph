# Conduct AUC and accuracy for main.py
from sklearn.metrics import roc_auc_score, accuracy_score

def compute_metrics(y_true, y_pred):
    """
    Compute the AUC and accuracy scores.
    
    Parameters:
    - y_true: Ground truth binary labels (1 for positive, 0 for negative).
    - y_pred: Predicted scores or probabilities for AUC calculation.
              Binary predictions (0 or 1) can be used for accuracy calculation.
              
    Returns:
    - A dictionary containing the AUC and Accuracy scores.
    """
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # Convert probabilities to binary predictions (0 or 1) if necessary
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    return auc, accuracy
