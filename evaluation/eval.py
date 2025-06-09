from transformers import Trainer, TrainingArguments
import datasets
from configs import configs
from configs.configs import to_bool
from data.raw import get_dataset
import torch
from vlm.utils_vlm import BatchDataCollator, get_vlm


def main():

    torch.set_default_dtype(torch.float16)

    config = configs.load_configs()["TRAIN"]

    if int(config["local_rank"]) != -1:
        torch.cuda.set_device(int(config["local_rank"]))
        print(config["local_rank"])

    if to_bool(config["load_dataset"]):

        dataset_test = datasets.load_from_disk(config['local_train_path'])
    else:

        dataset_test = get_dataset(config['test_annotations_path'], config['test_questions_path'], float(config['test_p']))

        dataset_test = dataset_test.add_column("split", ["test"] * len(dataset_test))

        dataset_train = dataset_test.add_column("img_path", [config['train_img_path']] * len(dataset_test))

        dataset_train.save_to_disk(config['local_test_path'])

    processor, vlm_model, vlm_config = get_vlm(config)
    processor.push_to_hub(config["output_dir"])
    vlm_config.push_to_hub(config["output_dir"])

    data_collator_batch = BatchDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy="no",  # Evaluate at the end of each epoch
        save_strategy="steps",
        save_steps=len(dataset_train),
        torch_empty_cache_steps=50,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["eval_batch_size"]),
        num_train_epochs=int(config["num_train_epochs"]),
        optim=config["optim"],
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        logging_steps=1,
        logging_dir="./logs",
        save_total_limit=1,
        fp16=True,
        fp16_full_eval=True,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=int(config["eval_accumulation_steps"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        batch_eval_metrics=True,
        dataloader_num_workers=8,

    )

    trainer = Trainer(
        model=vlm_model,
        args=training_args,
        train_dataset=dataset_test,
        eval_dataset=dataset_test,
        data_collator=data_collator_batch,
    )

    trainer.eval()


if __name__ == "__main__":
    main()