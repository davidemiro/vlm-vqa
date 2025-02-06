from sklearn.metrics import accuracy_score

def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)  # Get the index of the maximum probability
    accuracy = accuracy_score(labels, preds)
    return {"eval_accuracy": accuracy}