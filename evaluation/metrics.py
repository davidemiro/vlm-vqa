import evaluate

accuracy = evaluate.load("accuracy")


def compute_accuracy(p, compute_result=False):
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
        return mean

    else:

        predictions, labels = p
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = predictions.argmax(axis=-1)
        batch_accuracy = accuracy.compute(references=labels, predictions=preds)
        with open("store_values", 'a') as file:
            file.write(f"{batch_accuracy}\n")

    return batch_accuracy