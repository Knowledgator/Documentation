# Dataset Preparation

## Dataset Format Specification

GLiNER models require training data in a structured JSON format as shown below:

```json
{
  "tokenized_text": ["The", "Eiffel", "Tower", "is", "in", "Paris", "."],
  "ner": [
    [1, 3, "LANDMARK"],   // "Eiffel Tower"
    [5, 6, "LOCATION"]     // "Paris"
  ]
}
```

**Key Components:**
- `tokenized_text`: Array of individual tokens from your text
- `ner`: Array of entity annotations with start index, end index, and label

---

## Quickstart

Transform your raw data into GLiNER-ready format with this streamlined approach:

```python
from gliner import GLiNER
from gliner.data_processing import WordsSplitter, GLiNERDataset
from transformers import AutoTokenizer
import random
import json
import torch

# Initialize model components
model_id = 'knowledgator/gliner-multitask-large-v0.5'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GLiNER.from_pretrained(model_id)
model_config = model.config

# Configure compute device
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')

# Load and prepare your dataset
with open('your_dataset.json', 'r') as f:
  data = json.load(f)
random.shuffle(data)
train_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]

# Setup text processing
words_splitter = WordsSplitter(model_config.words_splitter_type)

# Create GLiNER dataset objects
train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)
```

## [numind/NuNER](https://huggingface.co/datasets/numind/NuNER) preparation example script
```python
from datasets import load_dataset
import re
import ast
import json
from tqdm import tqdm


def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def process_entities(dataset):
    """Processes entities in the dataset to extract tokenized text and named entity spans."""
    all_data = []
    for el in tqdm(dataset["entity"]):
        try:
            tokenized_text = tokenize_text(el["input"])
            parsed_output = ast.literal_eval(el["output"])
            entity_texts, entity_types = zip(*[i.split(" <> ") for i in parsed_output])

            entity_spans = []
            for j, entity_text in enumerate(entity_texts):
                entity_tokens = tokenize_text(entity_text)
                matches = []
                for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                    if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                        matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
                if matches:
                    entity_spans.extend(matches)

        except Exception as e:
            continue

        all_data.append({"tokenized_text": tokenized_text, "ner": entity_spans})
    return all_data


def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    dataset = load_dataset("numind/NuNER")
    processed_data = process_entities(dataset)

    save_data_to_file(processed_data, 'nuner_train.json')

    print("dataset size:", len(processed_data))
```