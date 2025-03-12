import evaluate
import torch
import time

accuracy = evaluate.load("accuracy")


def compute_accuracy(p, compute_result=False):
    t = time.time()
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

            mean = total_sum / count
        return {"accuracy" : mean}

    else:
        

        predictions, labels = p
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy().flatten()
        preds = predictions.argmax(axis=-1)
        preds = preds.flatten()

        batch_accuracy = accuracy.compute(references=labels, predictions=preds)["accuracy"]
        with open("store_values", 'a') as file:
            file.write(f"{batch_accuracy}\n")

        print("Time {}".format(time.time() - t))

    return {"accuracy": batch_accuracy}