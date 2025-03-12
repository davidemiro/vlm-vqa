from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from configs import configs
from data.raw import get_dataset
import torch
from evaluation.metrics import compute_accuracy
from vlm.utils_vlm import BatchDataCollator, get_vlm
import multiprocessing


def main():

    config = configs.load_configs()["TRAIN"]

    if int(config["local_rank"]) != -1:
        torch.cuda.set_device(int(config["local_rank"]))
        print(config["local_rank"])

    dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
    dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])

    dataset_train = dataset_train.add_column("split", ["train"] * len(dataset_train))
    dataset_val = dataset_val.add_column("split", ["val"] * len(dataset_val))

    dataset_train = dataset_train.add_column("img_path", [config['train_img_path']] * len(dataset_train))
    dataset_val = dataset_val.add_column("img_path", [config['val_img_path']] * len(dataset_val))

    processor, vlm_model, vlm_config = get_vlm(config)
    processor.push_to_hub(config["output_dir"])
    vlm_config.push_to_hub(config["output_dir"])

    if bool(config["lora"]):
        lora_config = LoraConfig(
            r=int(config['lora_rank']),
            lora_alpha=int(config['lora_alpha']),
            lora_dropout=float(config['lora_dropout']),
            bias=config['lora_bias'],
            task_type="CAUSAL_LM"
        )
        lora_model = get_peft_model(vlm_model, lora_config)

        lora_model.print_trainable_parameters()
        vlm_model = lora_model

    data_collator_batch = BatchDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="steps",  # Evaluate at the end of each epoch
        save_strategy="steps",
        eval_steps=8,
        save_steps=8,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=8,
        num_train_epochs=int(config["num_train_epochs"]),
        optim=config["optim"],
        push_to_hub=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=8,
        logging_dir="./logs",
        gradient_checkpointing=True,
        bf16=True,
        bf16_full_eval=True,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=8,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        batch_eval_metrics=True

    )

    trainer = Trainer(
        model=vlm_model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=data_collator_batch,
        compute_metrics=compute_accuracy,
    )

    trainer.train()


if __name__ == "__main__":
    main()







