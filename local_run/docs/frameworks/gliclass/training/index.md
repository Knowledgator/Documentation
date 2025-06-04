# Training
## Quickstart

### Installation

To install GLiClass, run the following command:

```bash
pip install gliclass
```

## Base Training Script {#base-training-script}

### Load pretrained model
```python
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

model_name = "knowledgator/gliclass-small-v1.0"
model = GLiClassModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
### Define Dataset

The dataset for training GLiClass must contain `text` field, which represents text for classification, `all_labels` field, whic stands for all labels to classify from and `true_labels` which will represent the correct labels for given text. For more info about datasets please visit our [datasets page](../prepared-datasets/index.mdx)

```python
from gliclass.data_processing import GLiClassDataset, DataCollatorWithPadding
data = [
    {
        "text": "A new machine learning platform automates complex data workflows but faces integration issues.",
        "all_labels": ["AI", "automation", "data_analysis", "usability", "integration"],
        "true_labels": ["AI", "integration", "automation"]
    }
]
train_dataset = GLiClassDataset(
    data,
    tokenizer,
    max_length= 1024, 
    problem_type= 'multi_label_classification', 
)

# Data collator
data_collator = DataCollatorWithPadding(device=device)
```
<details>
   <summary>Expected Output</summary>
   ```bash
   Total labels:  5
   ```
</details>

### Define functions for metrics computation
```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics_multi_label(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    preds = (predictions > 0.5).astype(int)
    labels = np.where(labels>0.5, 1, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

```

### Train the model
```python
from gliclass.training import TrainingArguments, Trainer

# Training arguments
training_args = TrainingArguments(
    output_dir="my-awesome-gliclass-model",
    learning_rate=1e-5,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="cosine",
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    use_cpu = False,
    report_to="none",
    fp16=False,  # Set to True if you want to use mixed precision training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_multi_label 
)

# Run training
trainer.train()
```

### Full Base Training Script <sup><a href="https://github.com/Knowledgator/GLiClass/blob/main/train.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

The following script could be used both for training from scratch and fine-tuning pretrained model:

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig

import random
import torch

from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset

def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    if args.problem_type == 'single_label_classification':
        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    elif args.problem_type == 'multi_label_classification':
        predictions = predictions.reshape(-1)
        preds = (predictions > 0.5).astype(int)
        labels = np.where(labels>0.5, 1, 0)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    else:
        raise NotImplementedError(f"{args.problem_type} is not implemented.")

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name, focal_loss_alpha=args.focal_loss_alpha,
                                                                focal_loss_gamma=args.focal_loss_gamma)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        if args.label_model_name is not None:
            label_model_config = AutoConfig.from_pretrained(args.label_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            label_model_name=args.label_model_name,
            label_model_config=label_model_config,
            class_token_index=len(tokenizer),
            text_token_index=len(tokenizer)+1,
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            contrastive_loss_coef=args.contrastive_loss_coef,
            normalize_features=args.normalize_features,
            extract_text_features=args.extract_text_features,
            architecture_type=args.architecture_type,
            prompt_first=args.prompt_first,
            squeeze_layers=args.squeeze_layers,
            shuffle_labels=args.shuffle_labels
        )

        model = GLiClassModel(glicalss_config, from_pretrained=True)

        if args.architecture_type in  {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder'}:
            new_words = ["<<LABEL>>", "<<SEP>>"]
            tokenizer.add_tokens(new_words, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    if model.config.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)
    else:
        labels_tokenizer = None

    model.config.problem_type = args.problem_type

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    train_dataset = GLiClassDataset(train_data, tokenizer, args.max_length, 
                                    args.problem_type, args.architecture_type, 
                                    args.prompt_first, labels_tokenizer=labels_tokenizer)
    test_dataset = GLiClassDataset(test_data, tokenizer, args.max_length, args.problem_type, 
                                        args.architecture_type, args.prompt_first,
                                        labels_tokenizer = labels_tokenizer)

    data_collator = DataCollatorWithPadding(device=device)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_steps = args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers = args.num_workers,
        logging_steps=100,
        use_cpu = False,
        report_to="none",
        fp16=args.fp16,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default= None)
    parser.add_argument('--encoder_model_name', type=str, default = 'microsoft/deberta-v3-small')
    parser.add_argument('--label_model_name', type=str, default = "BAAI/bge-small-en-v1.5")
    parser.add_argument('--save_path', type=str, default = 'models/')
    parser.add_argument('--data_path', type=str, default = 'data/zero-cats.json')
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='avg')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--architecture_type', type=str, default='uni-encoder')
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--prompt_first', type=bool, default=True)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--squeeze_layers', type=bool, default=False)
    parser.add_argument('--shuffle_labels', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--others_lr', type=float, default=3e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--focal_loss_alpha', type=float, default=-1)
    parser.add_argument('--focal_loss_gamma', type=float, default=-1)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    args = parser.parse_args()

    main(args)

```

## RL Training Script {#rl-training-script}

The GLiClass framework also supports Reinforcement learning, you can start training models using it with just a couple of changes to your training script.

### Load pretrained model
**This step leaves unchanged**
```python
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

model_name = "knowledgator/gliclass-small-v1.0"
model = GLiClassModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Initialize RL training components
```python
from transformers import AutoModelForSequenceClassification
from gliclass.pipeline import ZeroShotClassificationPipeline

# Value model for advantage estimation
value_model = AutoModelForSequenceClassification.from_pretrained(model.config.encoder_model_name, num_labels=1)
value_model.resize_token_embeddings(len(tokenizer))

# Reference model for baseline comparisons
refrence_model = GLiClassModel.from_pretrained(model_name) # for most cases you may use the same model as reference model
reference_tokenizer = AutoTokenizer.from_pretrained(model_name)
reference_pipe = ZeroShotClassificationPipeline(refrence_model, reference_tokenizer, 
                                                        classification_type='multi-label', 
                                                        progress_bar=False, device=device)
```

### Define Dataset

**This step leaves unchanged**

```python
from gliclass.data_processing import GLiClassDataset, DataCollatorWithPadding
data = [
    {
        "text": "A new machine learning platform automates complex data workflows but faces integration issues.",
        "all_labels": ["AI", "automation", "data_analysis", "usability", "integration"],
        "true_labels": ["AI", "integration", "automation"]
    }
]
train_dataset = GLiClassDataset(
    data,
    tokenizer,
    max_length= 1024, 
    problem_type= 'multi_label_classification', 
)

# Data collator
data_collator = DataCollatorWithPadding(device=device)
```
<details>
   <summary>Expected Output</summary>
   ```bash
   Total labels:  5
   ```
</details>

### Define functions for metrics computation

**This step leaves unchanged**

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics_multi_label(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    preds = (predictions > 0.5).astype(int)
    labels = np.where(labels>0.5, 1, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
```

### Define reward function
```python
def default_f1_reward(
    probs: torch.Tensor,
    actions: torch.Tensor,
    original_targets: torch.Tensor,
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    A variant that extracts list-of-indices sets and then calculates
    the F1 score in a classical manner. Returns shape (N, 1).
    
    Args:
        probs:              (N, T) Tensor of probabilities (not used here but left for interface consistency).
        actions:            (N, T) Tensor of predicted labels in {0, 1}.
        original_targets:   (N, T) Tensor of ground-truth labels in {0, 1}.
        valid_mask:         (N, T) Tensor indicating which positions are valid (1) vs. invalid (0).

    Returns:
        f1_scores: (N, 1) Tensor containing the F1 score for each row.
    """
    N = actions.shape[0]
    f1_scores = []

    for i in range(N):
        # Filter valid positions
        valid_preds_i = actions[i] * valid_mask[i]
        valid_targets_i = original_targets[i] * valid_mask[i]

        # Get the set of indices where we predicted 1
        predicted_set = set((valid_preds_i == 1).nonzero(as_tuple=True)[0].tolist())
        # Get the set of indices where the ground truth is 1
        target_set = set((valid_targets_i == 1).nonzero(as_tuple=True)[0].tolist())

        # Compute intersection
        intersection = predicted_set.intersection(target_set)

        # Precision
        if len(predicted_set) > 0:
            precision = len(intersection) / len(predicted_set)
        else:
            precision = 0.0

        # Recall
        if len(target_set) > 0:
            recall = len(intersection) / len(target_set)
        else:
            recall = 0.0

        # F1 score
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)
    
    # Convert list to tensor shape (N, 1)
    f1_scores = torch.tensor(f1_scores, dtype=torch.float).unsqueeze(-1)
    return f1_scores.detach().to(probs.device)
```

### Train the model with RLTrainer
```python
from gliclass.training import RLTrainerConfig, RLTrainer

training_args = RLTrainerConfig(
    output_dir="my-awesome-rl-gliclass-model",
    learning_rate=1e-5,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="cosine",
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    use_cpu = False,
    report_to="none",
    fp16=False, 
    cliprange=0.2,
    num_rl_iters=2
    )

trainer = RLTrainer(
    model=model,
    value_model=value_model, 
    reference_model=reference_pipe,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_multi_label,
    reward_components={
        'micro_f1': default_f1_reward,
    },
)

trainer.train()

```
:::important
To avoid `AttributeError` during run in notebooks add following lines after initializing trainer:
```python
trainer = RLTrainer(
    model=model,
    ...
)

from transformers.utils.notebook import NotebookProgressCallback
trainer.remove_callback(NotebookProgressCallback)

trainer.train()
```
:::

### Full RL Training Script <sup><a href="https://github.com/Knowledgator/GLiClass/blob/main/train_rl.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import random
import torch

from gliclass import GLiClassModelConfig, GLiClassModel, ZeroShotClassificationPipeline
from gliclass.training import TrainingArguments, Trainer, RLTrainerConfig, RLTrainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset
from gliclass.utils import default_f1_reward

def accuracy_reward(probs, actions, targets, valid_mask):
    probs = probs * valid_mask
    predicts = torch.argmax(probs, dim=-1)
    true_labels = torch.argmax(targets, dim=-1)
    correct = (predicts == true_labels).float().unsqueeze(1)
    return correct

def recall_reward(
    probs: torch.Tensor,
    actions: torch.Tensor,
    original_targets: torch.Tensor,
    valid_mask: torch.Tensor
) -> torch.Tensor:
    valid_preds = actions * valid_mask
    valid_targets = original_targets * valid_mask

    TP = torch.sum((valid_preds * valid_targets), dim=-1)
    FN = torch.sum(((1 - valid_preds) * valid_targets), dim=-1)

    eps = 1e-8
    recall = TP / (TP + FN + eps)
    return recall.detach().unsqueeze(1)

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name, focal_loss_alpha=args.focal_loss_alpha,
                                                                focal_loss_gamma=args.focal_loss_gamma)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        if args.label_model_name is not None:
            label_model_config = AutoConfig.from_pretrained(args.label_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            label_model_name=args.label_model_name,
            label_model_config=label_model_config,
            class_token_index=len(tokenizer),
            text_token_index=len(tokenizer)+1,
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            labels_smoothing=args.labels_smoothing,
            entropy_beta=args.entropy_beta,
            kl_beta=args.kl_beta,
            contrastive_loss_coef=args.contrastive_loss_coef,
            normalize_features=args.normalize_features,
            extract_text_features=args.extract_text_features,
            architecture_type=args.architecture_type,
            prompt_first=args.prompt_first,
            squeeze_layers=args.squeeze_layers
        )

        glicalss_config.problem_type = args.problem_type

        model = GLiClassModel(glicalss_config, from_pretrained=True)

        if args.architecture_type in  {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder'}:
            new_words = ["<<LABEL>>", "<<SEP>>"]
            tokenizer.add_tokens(new_words, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

    if args.set_value_model:
        value_model = AutoModelForSequenceClassification.from_pretrained(model.config.encoder_model_name, num_labels=1)
        value_model.resize_token_embeddings(len(tokenizer))
    else:
        value_model = None

    if args.reference_model is not None:
        refrence_model = GLiClassModel.from_pretrained(args.reference_model)
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
        reference_pipe = ZeroShotClassificationPipeline(refrence_model, reference_tokenizer, 
                                                                classification_type='multi-label', 
                                                                progress_bar=False, device=device)
    else:
        reference_pipe = None

    if args.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(args.label_model_name)
    else:
        labels_tokenizer = None

    model.to(device)
        
    with open(args.data_path, 'r') as f:
        data = json.load(f)[:]
    init_ld = len(data)*1

    print('Dataset size:', len(data))
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    train_dataset = GLiClassDataset(train_data, tokenizer, args.max_length, 
                                    args.problem_type, args.architecture_type, 
                                    args.prompt_first, labels_tokenizer=labels_tokenizer)
    test_dataset = GLiClassDataset(test_data, tokenizer, args.max_length, args.problem_type, 
                                        args.architecture_type, args.prompt_first,
                                        labels_tokenizer = labels_tokenizer)

    data_collator = DataCollatorWithPadding(device=device)

    training_args = RLTrainerConfig(
        output_dir=args.save_path,
        learning_rate=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_steps = args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers = args.num_workers,
        logging_steps=100,
        use_cpu = False,
        report_to="none",
        fp16=args.fp16,
        cliprange=args.clip_range,
        num_rl_iters=args.num_rl_iters
        )

    trainer = RLTrainer(
        model=model,
        value_model=value_model, 
        reference_model=reference_pipe,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        reward_components={
            'micro_f1': default_f1_reward,
        },
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default= "knowledgator/gliclass-modern-base-v2.0-init")
    parser.add_argument('--encoder_model_name', type=str, default = 'microsoft/deberta-v3-small')
    parser.add_argument('--label_model_name', type=str, default = "BAAI/bge-small-en-v1.5")
    parser.add_argument('--reference_model', type=str, default = None)
    parser.add_argument('--set_value_model', type=bool, default = True)
    parser.add_argument('--save_path', type=str, default = 'models/')
    parser.add_argument('--data_path', type=str, default = 'data/zero-cats.json')
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='avg')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--architecture_type', type=str, default='uni-encoder')
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--prompt_first', type=bool, default=True)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--squeeze_layers', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--encoder_lr', type=float, default=2e-6)
    parser.add_argument('--others_lr', type=float, default=3e-6)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--focal_loss_alpha', type=float, default=-1)
    parser.add_argument('--focal_loss_gamma', type=float, default=-1)
    parser.add_argument('--labels_smoothing', type=float, default=-1)
    parser.add_argument('--entropy_beta', type=float, default=-1)
    parser.add_argument('--kl_beta', type=float, default=0.1)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--num_rl_iters', type=int, default=2)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--save_steps', type=int, default=300)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    args = parser.parse_args()

    main(args)
```

## General recommendations

