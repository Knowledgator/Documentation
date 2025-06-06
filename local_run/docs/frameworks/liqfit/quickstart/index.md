# Quickstart

Welcome to the LiqFit Framework Quickstart Guide! This document will help you get started with the basics of using LiqFit.


## Installation

To install LiqFit, run the following command:

```bash
python -m pip install liqfit
```
Unlike SetFit, LiqFit works with any encoder or encoder-decoder model not just the Sentence Transformer model.

You can use your training function or just use Huggingface's Trainer which exists in the Transformers library.

## Training using the HuggingFace model:
1. Initialize your model
```python
from transformers import AutoModel, AutoTokenizer

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```
2. Initialize NLIDataset and load your NLI dataset from huggingface, NLI stands for Natural Language Inference. We will use the emotion dataset from Hugging Face
```python
from liqfit.datasets import NLIDataset
from datasets import load_dataset

emotion_dataset = load_dataset("dair-ai/emotion")
test_dataset = emotion_dataset['test']
classes = test_dataset.features["label"].names
# classes is a list that contains the following:
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


N = 8 # take few examples
train_dataset = emotion_dataset['train'].shuffle(seed=41).select(range(len(classes)*N))

nli_train_dataset = NLIDataset.load_dataset(train_dataset, classes=classes)
nli_test_dataset = NLIDataset.load_dataset(test_dataset, classes=classes)
```
3. Wrap your model with LiqFitModel, this will be useful if you want to pass your loss function or your downstream head if your backbone model does not have one (will show it below).
```python
from liqfit.modeling import LiqFitModel
from liqfit.losses import FocalLoss

backbone_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall')

loss_func = FocalLoss(multi_target=True)

model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)
```
4. Initialize your collator
```python
from liqfit.collators import NLICollator
data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)
```
5. Train your model
```python
training_args = TrainingArguments(
    output_dir='comprehendo',
    learning_rate=3e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=9,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_steps = 5000,
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
