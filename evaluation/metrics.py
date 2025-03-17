import evaluate
import os
accuracy = evaluate.load("accuracy")


def compute_f1_score(p, compute_result=False):
    if compute_result:
        total_sum = 0
        count = 0

        with open("store_values", 'r') as file:
            for line in file:
                try:
                    
                    value = float(line.strip())
                    total_sum += value
                    count += 1
                    
                except ValueError:
                    continue
            if count == 0:
                return 0

            os.remove("store_values")

            mean = total_sum / count
        return {"accuracy" : mean}

    else:

        predictions, labels = p
        predictions = predictions
        labels = labels.flatten()
        predictions = predictions.argmax(axis=-1).flatten()

        batch_accuracy = accuracy.compute(references=labels, predictions=predictions, average="macro")["f1"]
        with open("store_values", 'a') as file:
            file.write(f"{batch_accuracy}\n")

    return {"accuracy": batch_accuracy}