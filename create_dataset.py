from dataset import get_dataset
import configs
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoImageProcessor
import vlm
import multiprocessing

config = configs.load_configs()["TRAIN"]

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])

tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
tokenizer = vlm.vlm_tokenizer(tokenizer)

vlm_config = AutoConfig.from_pretrained(config['name'], token=config['token'])
vlm_config.visual_embed_dim = 768
vlm_config.pad_token_id = tokenizer.pad_token_id
vlm_config.image_token_id = tokenizer.convert_tokens_to_ids('<image>')

vit_config = AutoConfig.from_pretrained(config['vit_name'])
num_patches = (vit_config.patch_size + 2) ** 2 + 1
processor = AutoImageProcessor.from_pretrained(config["vit_name"])

data_collator_train = vlm.VLMDataCollator(tokenizer, int(config['text_length']), processor, num_patches, config['train_img_path'], split='train')
data_collator_val = vlm.VLMDataCollator(tokenizer, int(config['text_length']), processor, num_patches, config['val_img_path'], split='val')

dataset_train = dataset_train.map(data_collator_train, batched=False, num_proc=multiprocessing.cpu_count())
dataset_val = dataset_val.map(data_collator_val, batched=False, num_proc=multiprocessing.cpu_count())

dataset_train.save_to_disk(config['train_path'])
dataset_val.save_to_disk(config['val_path'])