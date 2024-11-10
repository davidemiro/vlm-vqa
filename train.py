from vlm.modelling_vlm import VLMGemma2ForCausalLM
from vlm.configuration_vlm import VLMConfig
from vlm.utils_vlm import BatchDataCollator
from transformers import Trainer, TrainingArguments
from configs import configs
import datasets


config = configs.load_configs()["TRAIN"]

dataset_train = datasets.load_from_disk(config['train_path'])
dataset_val = datasets.load_from_disk(config['val_path'])

vlm_config = VLMConfig()
vlm_model = VLMGemma2ForCausalLM.from_pretrained(config['name'], config=vlm_config, token=config['token'])


data_collator_batch = BatchDataCollator()

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    per_device_train_batch_size=int(config["batch_size"]),
    num_train_epochs=int(config["num_train_epochs"]),
    optim=config["optim"],
    push_to_hub=True,

)

trainer = Trainer(
    model=vlm_model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator_batch,
)


trainer.train()



