# 💧 LiqFit

**LiqFit** is an easy-to-use framework for few-shot learning of cross-encoder models. These models are trained to determine whether two statements entail, contradict each other, or are neutral — a task setup applicable to various information extraction problems, such as text classification, named entity recognition, and question answering. With LiqFit, competitive results can be achieved using as few as 8 examples per label.

For detailed information and access to the code, visit the [GitHub repository](https://github.com/Knowledgator/LiqFit).

---

## Key Features

- 🔢 **Few-shot capable**  
  Achieve high performance with only 8 examples per label. Significantly improves default zero-shot classifiers.

- 📝 **Task-agnostic inference setting**  
  Leverages natural language inference (NLI) to generalize across various information extraction tasks, including NER and QA.

- 🌈 **Generalizes to unseen classes**  
  Models maintain the ability to classify labels not present in the training set, thanks to pre-finetuning on large NLI/classification corpora.

- ⚙️ **Supports multiple cross-encoder types**  
  Compatible with standard, binary, and encoder-decoder cross-encoder architectures.

- ⚖️ **Robust to class imbalance**  
  Normalization techniques ensure stability and performance even with unbalanced datasets.

- 🏷️ **Multi-label classification support**  
  Designed for both multi-class and multi-label classification tasks.

---
## Benchmarks:
| Model & examples per label | Emotion | AgNews | SST5 |
|-|-|-|-|
| Comprehend-it/0 | 56.60 | 79.82 | 37.9 |  
| Comprehend-it/8 | 63.38 | 85.9 | 46.67 |
| Comprehend-it/64 | 80.7 | 88 | 47 |
| SetFit/0 | 57.54 | 56.36 | 24.11 |
| SetFit/8 | 56.81 | 64.93 | 33.61 |  
| SetFit/64 | 79.03 | 88 | 45.38 |

## Limitations

- 🤔 **Computational overhead**  
  Requires `N` forward passes through the transformer for `N` labels.
