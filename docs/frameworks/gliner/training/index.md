# Training
## Quickstart
### Installation

To install GLiNER, run the following command:

```bash
pip install gliner
```
### Load pretrained model
```python
from transformers import AutoTokenizer
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator

# Load model and tokenizer
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-v1.0")
tokenizer = AutoTokenizer.from_pretrained(model.config.model_name)
```

### Define Dataset

The dataset for training GLiNER should contain tokenized text and the start and end positions of each entity in the text.
```python
# Dummy training data
train_data = [
    {"tokenized_text": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."], "ner": [[0, 1, "Person"], [5, 5, "Location"]]}
]
```
### Train the model
```python
# Training arguments
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    report_to="none",
)

# Data collator
data_collator = DataCollator(
    config=model.config,
    data_processor=model.data_processor,
    prepare_labels=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Run training
trainer.train()
```

## Base Training Script {#training-script}

The following script could be used both for training from scratch and fine-tuning pretrained model:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import random
import json

from transformers import AutoTokenizer
import torch

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= "configs/config.yaml")
    parser.add_argument('--log_dir', type=str, default = 'models/')
    parser.add_argument('--compile_model', type=bool, default = False)
    parser.add_argument('--freeze_language_model', type=bool, default = False)
    parser.add_argument('--new_data_schema', type=bool, default = False)
    args = parser.parse_args()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    with open(config.train_data, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    #shuffle
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')


    if config.prev_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.prev_path)
        model = GLiNER.from_pretrained(config.prev_path)
        model_config = model.config
    else:
        model_config = GLiNERConfig(**vars(config))
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
        words_splitter = WordsSplitter(model_config.words_splitter_type)

        model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)

        if not config.labels_encoder:
            model_config.class_token_index=len(tokenizer)
            tokenizer.add_tokens([model_config.ent_token, model_config.sep_token], special_tokens=True)
            model_config.vocab_size = len(tokenizer)
            model.resize_token_embeddings([model_config.ent_token, model_config.sep_token], 
                                        set_class_token_index = False,
                                        add_tokens_to_tokenizer=False)

    if args.compile_model:
        torch.set_float32_matmul_precision('high')
        model.to(device)
        model.compile_for_training()
        
    if args.freeze_language_model:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(False)
    else:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(True)

    if args.new_data_schema:
        train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
        test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)
        data_collator = DataCollatorWithPadding(model_config)
    else:
        train_dataset = train_data
        test_dataset = test_data
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        focal_loss_gamma=config.loss_gamma,
        focal_loss_alpha=config.loss_alpha,
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        evaluation_strategy="epoch",
        save_steps = config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers = 8,
        use_cpu = False,
        report_to="none",
        bf16=True,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
```
## General recommendations
To fine-tune pretrained models, you should set parameter `prev_path` to the name of the model you want to fine-tune. The model configuration would be implicitly loaded from the checkpoint and you cannot change it during the fine-tuning with base training script, [see which parameters would be inherited](../components--configs/#training-configuration).

If you want to train model from scratch, `prev_path` should be set to `null`. If parameter `labels_encoder` is not `null`, the architecture would be turned into a bi-encoder.

## Fine-tuning Uni-Encoder GLiNER
Here is an example how to fine-tune pretrained uni-encoder GLiNER model. To fine-tune a specific model, define the name of the model with `prev_path` argument. For example, we want to fine-tune [`knowledgator/gliner-multitask-large-v0.5`](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5).

### Define config.yaml
```yaml config.yaml
model_name: null

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 5000
warmup_ratio: 0.1
scheduler_type: "cosine"

# loss function
loss_alpha: -1 # focal loss alpha, if -1, no focal loss
loss_gamma: 0 # focal loss gamma, if 0, no focal loss
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate and weight decay Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01

max_grad_norm: 1.0

# Directory Paths
root_dir: span_gliner_logs
train_data: "data.json" # see https://github.com/urchade/GLiNER/tree/main/data
val_data_dir: "none"
# "NER_datasets": val data from the paper can be obtained from "https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view"

# Pretrained Model Path
# Use "none" if no pretrained model is being used
prev_path: "knowledgator/gliner-multitask-large-v0.5"

save_total_limit: 3 #maximum amount of checkpoints to save

# Advanced Training Settings
size_sup: -1
max_types: 25
shuffle_types: true
random_drop: true
max_neg_type_ratio: 1
max_len: 384
freeze_token_rep: false
```
### Run [training script](#training-script)
Once config.yaml is defined, run the training script.

```python
python train.py --config "config.yaml"
```

## Fine-tuning Bi-Encoder GLiNER
Here is an example how to fine-tune pretrained bi-encoder GLiNER model. To fine-tune a specific model, define the name of the model with `prev_path` argument. For example, we want to fine-tune [`knowledgator/modern-gliner-bi-base-v1.0`](https://huggingface.co/knowledgator/modern-gliner-bi-base-v1.0).

### Define config.yaml
```yaml config.yaml
model_name: null

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 5000
warmup_ratio: 0.1
scheduler_type: "cosine"

# loss function
loss_alpha: -1 # focal loss alpha, if -1, no focal loss
loss_gamma: 0 # focal loss gamma, if 0, no focal loss
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate and weight decay Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01

max_grad_norm: 1.0

# Directory Paths
root_dir: span_gliner_logs
train_data: "data.json" # see https://github.com/urchade/GLiNER/tree/main/data
val_data_dir: "none"
# "NER_datasets": val data from the paper can be obtained from "https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view"

# Pretrained Model Path
# Use "none" if no pretrained model is being used
prev_path: "knowledgator/modern-gliner-bi-base-v1.0"

save_total_limit: 3 #maximum amount of checkpoints to save

# Advanced Training Settings
size_sup: -1
max_types: 25
shuffle_types: true
random_drop: true
max_neg_type_ratio: 1
max_len: 384
freeze_token_rep: false
```
### Run [training script](#training-script)
Once config.yaml is defined, run the training script.

```python
python train.py --config "config.yaml"
```
## Training Uni-Encoder GLiNER from scratch
Here is an example how to train a uni-encoder GLiNER model from scratch. To fine-tune a specific model, define the name of the model with `prev_path` argument. For example, we want to fine-tune [`knowledgator/gliner-multitask-large-v0.5`](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5).

### Define config.yaml
```yaml config.yaml
# Model Configuration
model_name: microsoft/deberta-v3-base # Hugging Face model
labels_encoder:
name: "span level gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
fuse_layers: false
post_fusion_schema: ""
span_mode: markerV0

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# loss function
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate and weight decay Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01

max_grad_norm: 10.0

# Directory Paths
root_dir: gliner_logs
train_data: "data.json" #"data/nuner_train.json" # see https://github.com/urchade/GLiNER/tree/main/data
val_data_dir: "none"
# "NER_datasets": val data from the paper can be obtained from "https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view"

# Pretrained Model Path
# Use "none" if no pretrained model is being used
prev_path: null

save_total_limit: 3 #maximum amount of checkpoints to save

# Advanced Training Settings
size_sup: -1
max_types: 25
shuffle_types: true
random_drop: true
max_neg_type_ratio: 1
max_len: 386
freeze_token_rep: false
```
### Run [training script](#training-script)
Once config.yaml is defined, run the training script.

```python
python train.py --config "config.yaml"
```
## Training Bi-Encoder GLiNER from scratch
Here is an example how to fine-tune pretrained uni-encoder GLiNER model. To fine-tune a specific model, define the name of the model with `prev_path` argument. For example, we want to fine-tune [`knowledgator/gliner-multitask-large-v0.5`](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5).

### Define config.yaml
```yaml config.yaml
# Model Configuration
model_name: microsoft/deberta-v3-base # Hugging Face model
labels_encoder: sentence-transformers/all-MiniLM-L6-v2
name: "span level gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
fuse_layers: false
post_fusion_schema: ""
span_mode: markerV0

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# loss function
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate and weight decay Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01

max_grad_norm: 10.0

# Directory Paths
root_dir: gliner_logs
train_data: "data.json" #"data/nuner_train.json" # see https://github.com/urchade/GLiNER/tree/main/data
val_data_dir: "none"
# "NER_datasets": val data from the paper can be obtained from "https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view"

# Pretrained Model Path
# Use "none" if no pretrained model is being used
prev_path: null

save_total_limit: 3 #maximum amount of checkpoints to save

# Advanced Training Settings
size_sup: -1
max_types: 25
shuffle_types: true
random_drop: true
max_neg_type_ratio: 1
max_len: 386
freeze_token_rep: false
```
### Run [training script](#training-script)
Once config.yaml is defined, run the training script.

```python
python train.py --config "config.yaml"
```

## Evaluation

\\TODO Add info how to download eval data and run evaluation on standard datasets. 
To evaluate your model you could use GLiNER native evaluation method.

```python
# Assume model is defined
from GLiNER.evaluation import get_for_all_path

get_for_all_path(model, steps, log_dir, data_paths)