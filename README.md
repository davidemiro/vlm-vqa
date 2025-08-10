# Visual Language Model for Visual Question Answering

## Overview

This repo contains the implementation of a **Visual Language Model for Visual Question Answering (VLM-VQA)**.
It proposes the combination of **DINOv2** as vision encoder **and Gemma 2 (2B)** as language model, following the idea outlined in the research paper "PaliGemma: A versatile 3B VLM for transfer."
The model has been trained specifically for the Visual Question Answering task, demostrating competitive performance.
Moreover the implementation, in this repo you can find the code for the **multi-GPU** training with **deepspeed** and pytorch **Distributed Data Parallel (DDP)** and **Fully Shard Data Parallel (FSDP)**


## Architecture

- The vision encoder **DINOv2** DINOv2 is a self-supervised vision transformer, providing high-performance visual features that can be directly employed on a variety of visual tasks.
  A **Vision Transformer (ViT)** is a transformer encoder model pretrained on large collection. The image are represented as a sequence of fixed-size patches.
  During the pretraining, the model learns the inner representation of images that can then be used to extract features useful for downstream tasks.
  In this case, the Vision Encoder computes the embeddings of the input images, the input of the language model.
  
- The language model **Gemma 2 (2B)** is a decoder only transformer available in english. The **Language Model** accept as input a text, as a sequence of tokens, and it generate the next token based on all the ones before it. In order to make the language model able to handle an image as input, it has been defined a **linear projector** that allows to project the visual features to the dimension of text ones.




## Training

The model has been trained using the tool **Trainer** of **Hugging Face** **Transformers** library.

| Parameter                  | Value             |
|----------------------------|-------------------|
| batch_size                 | 6                 |
| gradient_accumulation_steps| 10                |
| learning_rate              | 1e-5              |
| weight_decay               | 1e-4              |
| dropout                    | 0.1               |
| optim                      | adamw_bnb_8bit    |
| lr_scheduler_type          | cosine            |
| warmup_ratio               | 0.01              |
| max_grad_norm              | 1.0               |
| percentile_clipping        | 95                |
| block_wise                 | True              |
| num_train_epochs           | 1                 |
| text_length                | 64                |
| num_patches                | 257               |
| text_length                | 64                |
| visual_embed_dim           | 768               |

### Fully Shard Data Parallel

This model was trained in a **multi-GPU** environment using **Fully Sharded Data Parallel (FSDP)** with **Hugging Face** **Accelerate**.
Specifically, it has been used ** 4 x NVIDIA A10G **.
FSDP shards model parameters, optimizer states and gradients across all GPUs. It has been adopted the **TRANSFORMER_BASED_WRAP** that allows to select the layers to shard.
If you want to train the model you can use the following command:

```bash accelerate launch \ --config_file torch_distributed/fsdp2_config.yml \ train.py \ --token <hf_token> \ --configs torch_distributed/fsdp_configs.ini ``` 


## Evaluation

The model has been evaluated with **Holistic Evaluation of Language Models (HELM)**.
It has been evaluated on 1K instances of VQAv2 HELM benchmark reaching a **0.5 BLEU score**.


