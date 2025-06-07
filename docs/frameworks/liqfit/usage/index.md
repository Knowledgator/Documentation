# Usage
## Emotion Classification with LiqFit

This example demonstrates how to fine-tune a cross-encoder model using the LiqFit framework for multi-label emotion classification.

### Setup

Install the required packages:

```bash
pip install liqfit datasets transformers
```

### Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("emotion")
```

### Define Classes

```python
classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
```

### Prepare Data

```python
from liqfit.datasets import NLIDataset

train_dataset = NLIDataset(
    premises=dataset['train']['text'],
    hypotheses=classes,
    labels=dataset['train']['label'],
    multi_label=True
)

test_dataset = NLIDataset(
    premises=dataset['test']['text'],
    hypotheses=classes,
    labels=dataset['test']['label'],
    multi_label=True
)
```

### Initialize Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from liqfit.modeling import LiqFitModel
from liqfit.losses import FocalLoss

backbone = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall')
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')
loss_func = FocalLoss()

model = LiqFitModel(config=backbone.config, backbone=backbone, loss_func=loss_func)
```

### Define Data Collator

```python
from liqfit.collators import NLICollator

data_collator = NLICollator(tokenizer=tokenizer, max_length=128, padding=True, truncation=True)
```

### Set Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='emotion_model',
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_steps=1000,
    save_total_limit=2,
    remove_unused_columns=False,
)
```

### Train the Model

```python
from transformers import Trainer

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

## Evaluate the Model

```python
from sklearn.metrics import classification_report
from liqfit import ZeroShotClassificationPipeline

classifier = ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)

predictions = []
for example in dataset['test']:
    result = classifier(example['text'], classes, multi_label=True)
    predictions.append([label for label, score in result.items() if score > 0.5])

print(classification_report(dataset['test']['label'], predictions, target_names=classes))
```

---

For more details, refer to the [train_emotions_classifier.ipynb](https://github.com/Knowledgator/LiqFit/blob/main/notebooks/train_emotions_classifier.ipynb) notebook.


## Training with a Hugging Face Model

1. Initialize the Model and Tokenizer

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

2. Prepare the Dataset

Use the `emotion` dataset from Hugging Face and convert it into an NLI format using `NLIDataset`.

```python
from datasets import load_dataset
from liqfit.datasets import NLIDataset

emotion_dataset = load_dataset("dair-ai/emotion")
test_dataset = emotion_dataset['test']
classes = test_dataset.features["label"].names

N = 8  # Number of examples per class
train_dataset = emotion_dataset['train'].shuffle(seed=41).select(range(len(classes) * N))

nli_train_dataset = NLIDataset.load_dataset(train_dataset, classes=classes)
nli_test_dataset = NLIDataset.load_dataset(test_dataset, classes=classes)
```

3. Wrap the Model with LiqFitModel

```python
from liqfit.modeling import LiqFitModel
from liqfit.losses import FocalLoss

backbone_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall')
loss_func = FocalLoss(multi_target=True)

model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)
```

4. Initialize the Data Collator

```python
from liqfit.collators import NLICollator

data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)
```

5. Set Training Arguments and Train the Model

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='comprehendo',
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

---

## Training with a Custom Downstream Head

If your model outputs only the last hidden state without a classification head, you can define a custom downstream head.

```python
from transformers import AutoModel, AutoTokenizer
from liqfit.modeling import ClassClassificationHead, LiqFitModel

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

hidden_size = model.config.hidden_size
downstream_head = ClassClassificationHead(in_features=hidden_size, out_features=3)

wrapped_model_with_custom_head = LiqFitModel(model.config, model, head=downstream_head)
```

---

## Customizing Backbone Model Outputs

For advanced use cases, wrap your backbone model to customize its outputs before passing them to the downstream head.

```python
from transformers import AutoModel, AutoTokenizer
from liqfit.modeling import LiqFitBackbone, LiqFitModel, ClassClassificationHead
from liqfit.modeling.pooling import GlobalMaxPooling1D

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)

class BackboneWrapped(LiqFitBackbone):
    def __init__(self):
        model = AutoModel.from_pretrained(model_path)
        super().__init__(config=model.config, backbone=model)
        self.pooler = GlobalMaxPooling1D()

    def forward(self, x):
        output = self.backbone(x)
        last_hidden_state = output.last_hidden_state
        pooled_output = self.pooler(last_hidden_state)
        return pooled_output

my_wrapped_backbone = BackboneWrapped()
downstream_head = ClassClassificationHead(in_features=model.config.hidden_size, out_features=3)

wrapped_model_with_custom_head = LiqFitModel(my_wrapped_backbone.config, my_wrapped_backbone, head=downstream_head)
```

---

## Using `ClassificationHead` for Custom Loss Functions and Poolers

For flexibility in specifying different loss functions or custom poolers:

```python
from transformers import AutoModel, AutoTokenizer
from liqfit.modeling import ClassificationHead, LiqFitModel
from liqfit.losses import FocalLoss
from liqfit.modeling.pooling import GlobalMaxPooling1D

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

hidden_size = model.config.hidden_size
loss = FocalLoss()
pooler = GlobalMaxPooling1D()
downstream_head = ClassificationHead(in_features=hidden_size, out_features=3, loss_func=loss, pooler=pooler)

wrapped_model_with_custom_head = LiqFitModel(model.config, model, head=downstream_head)
```

---