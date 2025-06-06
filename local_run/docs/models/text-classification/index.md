# Text Classification Models

Knowledgator provides a collection of powerful models for **text classification** tasks, supporting both **zero-shot** and **few-shot** learning scenarios. These models are designed to handle a variety of use cases such as sentiment analysis, topic detection, intent classification, and more.

---

## Available Models

### 🔹 ComprehendIt
A DeBERTaV3-based model fine-tuned on NLI and classification datasets. Suitable for general-purpose zero-shot classification across multiple domains.

### 🔹 GLiClass
A lightweight model optimized for high-speed inference in low-resource environments, suitable for real-time applications and edge devices.

## Key Features

- Support for **multi-label** and **multi-class** classification.
- Optimized for **zero-shot inference** using NLI-style input formulation.
- Fine-tuning capabilities with the [LiqFit](https://github.com/Knowledgator/LiqFit) framework.
- Usable with Hugging Face's `pipeline` API or custom training setups.

## Use Cases

- Sentiment Analysis  
- Intent Recognition  
- Topic Classification  
- Content Moderation  
- Customer Feedback Analysis  

For detailed documentation and usage examples, visit [Knowledgator Docs](https://docs.knowledgator.com/).
