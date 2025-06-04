# Dataset Preparation

## Dataset Format Specification

GLiClass models require training data in a structured JSON format as shown below:

```json
[
  {
    "text": "Basketball players need excellent coordination and quick reflexes.",
    "all_labels": 
        [
            "sports",
            "science",
            "business"
        ],
    "true_labels": 
        [
            "sports"
        ]
  },
  ...
]
```
:::note
It is also possible to specify confidence scores explicitly:
```json
[
  {
    "text": "Basketball players need excellent coordination and quick reflexes.",
    "all_labels": 
        [
            "sports",
            "science",
            "business"
        ],
    "true_labels": 
        {
            "sports": 0.9
        }
  },
  ...
]
```
:::

## Quickstart

You can easily transform any oficial Knowledgator's [GLiClass datasets](../prepared-datasets/index.mdx) to requiered format using following script:

```python
import json
from datasets import load_dataset
def kg_dataset_to_json(dataset_name, split='train', output_path=None):
    output_path = output_path or f"{dataset_name.split('/')[-1]}.json"
    dataset = load_dataset(dataset_name, split=split)
    
    data_list = []
    for item in dataset:
        data_list.append({
            "text": item["text"],
            "all_labels": item["all_labels"], 
            "true_labels": item["true_labels"]
        })
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)
    
    print(f"Data saved to {output_path}")

kg_dataset_to_json("knowledgator/gliclass-v2.0", split="train")
```
