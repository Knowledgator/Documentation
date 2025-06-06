# Comprehend-it

[**Comprehend-it**](https://huggingface.co/knowledgator/comprehend_it-base) is a versatile NLP model developed by Knowledgator, designed for zero-shot and few-shot learning across various information extraction tasks. Built upon the DeBERTaV3-base architecture, it has been extensively trained on natural language inference (NLI) and diverse text classification datasets, enabling it to outperform larger models like BART-large-MNLI while maintaining a smaller footprint.

---

## Overview

- **Architecture**: Based on DeBERTaV3-base.
- **Model Size**: 184 million parameters.
- **Input Capacity**: Up to 3,000 tokens.
- **Languages Supported**: English.

---

## Supported Information Extraction Tasks

- Text Classification
- Reranking of Search Results
- Named Entity Recognition (NER)
- Relation Extraction
- Entity Linking
- Question Answering

---

## Benchmarking

Zero-shot F1 scores on various datasets:

| Model                | IMDB | AG_NEWS | Emotions |
|----------------------|------|---------|----------|
| BART-large-MNLI      | 0.89 | 0.6887  | 0.3765   |
| DeBERTa-base-v3      | 0.85 | 0.6455  | 0.5095   |
| **Comprehend-it**    | 0.90 | 0.7982  | 0.5660   |
| SetFit BAAI/bge-small-en-v1.5 | 0.86 | 0.5636  | 0.5754   |

*Note: All models were evaluated in a zero-shot setting without fine-tuning.*

---

## Usage

### Zero-Shot Classification with Hugging Face Pipeline

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="knowledgator/comprehend_it-base")

sequence = "One day I will see the world"
labels = ['travel', 'cooking', 'dancing']

result = classifier(sequence, labels)
print(result)
```

### Multi-Label Classification

```python
labels = ['travel', 'cooking', 'dancing', 'exploration']
result = classifier(sequence, labels, multi_label=True)
print(result)
```

### Manual Inference with PyTorch

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained('knowledgator/comprehend_it-base')
tokenizer = AutoTokenizer.from_pretrained('knowledgator/comprehend_it-base')

premise = "One day I will see the world"
hypothesis = "This example is travel."

inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt')
logits = model(inputs).logits

entail_contradiction_logits = logits[:, [0, 2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:, 1]
print(prob_label_is_true)
```

---

## Question Answering

Transform QA tasks into a multi-choice format:

```python
question = "What is the capital city of Ukraine?"
candidate_answers = ['Kyiv', 'London', 'Berlin', 'Warsaw']
result = classifier(question, candidate_answers)
print(result)
```

---

## Few-Shot Fine-Tuning with LiqFit
Install LiqFit via pip:

```bash
pip install liqfit
```

---

```python
from liqfit.modeling import LiqFitModel
from liqfit.losses import FocalLoss
from liqfit.collators import NLICollator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained('knowledgator/comprehend_it-base')
backbone_model = AutoModelForSequenceClassification.from_pretrained('knowledgator/comprehend_it-base')
loss_func = FocalLoss()

model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)
data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)

training_args = TrainingArguments(
    output_dir='comprehend_it_model',
    learning_rate=3e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=9,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_steps=5000,
    save_total_limit=3,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=nli_train_dataset,
    eval_dataset=nli_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```