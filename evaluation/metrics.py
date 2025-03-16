import evaluate
import os
f1_score = evaluate.load("f1")


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
        return {"f1_score" : mean}

    else:

        predictions, labels = p
        predictions = predictions
        labels = labels.flatten()
        predictions = predictions.argmax(axis=-1).flatten()

        batch_accuracy = f1_score.compute(references=labels, predictions=predictions)["f1"]
        with open("store_values", 'a') as file:
            file.write(f"{batch_accuracy}\n")

    return {"f1_score": batch_accuracy}