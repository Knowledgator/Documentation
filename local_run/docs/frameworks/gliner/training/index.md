# Training

## Fine-tuning

### Quickstart
```python
from transformers import AutoTokenizer
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator

# Load model and tokenizer
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-v1.0")
tokenizer = AutoTokenizer.from_pretrained(model.config.model_name)

# Dummy training data
train_data = [
    {"tokenized_text": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."], "ner": [[0, 1, "Person"], [5, 5, "Location"]]}
]

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