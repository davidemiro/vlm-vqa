import evaluate
import os
import torch
accuracy = evaluate.load("accuracy")


def compute_accuracy(p, compute_result=False):
    gpu_rank = torch.cuda.current_device()

    print("Computing accuracy... {}".format(compute_result))
    if compute_result:
        total_sum = 0
        count = 0

        with open("store_values_{}".format(gpu_rank), 'r') as file:
            for line in file:
                try:
                    
                    value = float(line.strip())
                    total_sum += value
                    count += 1
                    
                except ValueError:
                    continue
            if count == 0:
                return 0

            os.remove("store_values_{}".format(gpu_rank))


            mean = total_sum / count

            print("Mean {}: {}".format(gpu_rank, mean))
        return {"accuracy" : mean}

    else:

        predictions, labels = p
        predictions = predictions
        labels = labels.flatten()
        predictions = predictions.argmax(axis=-1).flatten()

        batch_accuracy = accuracy.compute(references=labels, predictions=predictions)["accuracy"]
        with open("store_values", 'a') as file:
            file.write(f"{batch_accuracy}\n")

    return {"accuracy": batch_accuracy}