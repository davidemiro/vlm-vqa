import evaluate
import os
import torch
import threading
accuracy = evaluate.load("accuracy")


def compute_accuracy_closure(input_lock, input_num_gpu):
    lock = input_lock
    num_gpu = input_num_gpu

    def compute_accuracy(p, compute_result=False):

        nonlocal num_gpu
        nonlocal lock

        print("Computing accuracy... {}".format(compute_result))
        if compute_result:
            total_sum = 0
            count = 0
            mean = 0

            with lock:

                for gpu_rank in range(num_gpu):

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

                    print("Mean : {}".format(mean))
            return {"eval_accuracy": mean}

        else:
            gpu_rank = torch.cuda.current_device()

            predictions, labels = p
            predictions = predictions
            labels = labels.flatten()
            predictions = predictions.argmax(axis=-1).flatten()

            batch_accuracy = accuracy.compute(references=labels, predictions=predictions)["accuracy"]
            with open("store_values_{}".format(gpu_rank), 'a') as file:
                file.write(f"{batch_accuracy}\n")

            return {"eval_accuracy": batch_accuracy}

    return compute_accuracy