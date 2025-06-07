# Quickstart

Welcome to the GLiClass Framework Quickstart Guide! This document will help you get started with the basics of using GLiClass.


## Installation

To install GLiClass, run the following command:

```bash
pip install gliclass
```

## Basic Usage

Here is a simple example to get started:

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device=device
)

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0]

for result in results:
    print(f"{result['label']} => {result['score']:.3f}")
```
<details>
    <summary>Expected Output</summary>
    ```bash  
    travel => 1.000  
    dreams => 1.000  
    sport => 1.000  
    science => 1.000  
    politics => 0.817  
    ```
</details>

## Next Steps

- Check out the [Examples](../examples/index.md) for more use cases.

Happy coding!