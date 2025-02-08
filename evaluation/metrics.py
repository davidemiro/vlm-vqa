import evaluate

clf_metrics = evaluate.load("accuracy")

def compute_accuracy(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)  # Get the index of the maximum probability
    return clf_metrics(labels, preds)