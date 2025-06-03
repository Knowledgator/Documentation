# Usage

## 🚀 Basic Use Case

After the installation of the GLiClass library, import the `GLiClassModel` class. Following this, you can load your chosen model with `GLiClassModel.from_pretrained` and utilize `ZeroShotClassificationPipeline` to classify your texts.

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize GLiClass model and tokenizer
model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

# Initialize Classification pipeline
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device=device
)

# Sample text for classification
text = """
One day I will see the world!
"""

# Labels for classification
labels = ["travel", "dreams", "sport", "science", "politics"]

# Perform classification
results = pipeline(text, labels, threshold=0.5)[0]

# Display predicted classes and their scores
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

## Retrieval-Augmented Classification (RAC)

With some models trained with retrieval-agumented classification, such as [`knowledgator/gliclass-base-v2.0-rac-init`](https://huggingface.co/knowledgator/gliclass-base-v2.0-rac-init) you can specify examples to improve classification accuracy:

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device=device
)

# Add an example for model to improve performance
example = {
    "text": "A new machine learning platform automates complex data workflows but faces integration issues.",
    "all_labels": ["AI", "automation", "data_analysis", "usability", "integration"],
    "true_labels": ["AI", "integration", "automation"]
}

text = "The new AI-powered tool streamlines data analysis but has limited integration capabilities."
labels = ["AI", "automation", "data_analysis", "usability", "integration"]

results = pipeline(text, labels, threshold=0.1, rac_examples=[example])[0]

for predict in results:
    print(f"{predict['label']} => {predict['score']:.3f}")
```
<details>
    <summary>Expected Output</summary>
    ```bash
    AI => 1.000
    automation => 1.000
    data_analysis => 1.000
    usability => 1.000
    integration => 1.000
    ```
</details>

## 🛠️Pipelines {#pipelines}
- **Sentiment Analysis**: Rapidly classify texts as positive, negative, or neutral.
- **Document Classification**: Efficiently organize and categorize large document collections.
- **Search Results Re-ranking**: Improve relevance and precision by reranking search outputs.
- **News Categorization**: Automatically tag and organize news articles into predefined categories.
- **Fact Checking**: Quickly validate and categorize statements based on factual accuracy.

We have prepared high-level classes that simplify working with GLiClass

### Zero-Shot Classification

The `ZeroShotClassificationPipeline` is a pipeline for text classification tasks based on the GLiClass model. It evaluates input text against a set of predefined labels, supporting both single-label and multi-label classification.

1. **Initialize the Pipeline**  
    Load a pretrained model and initialize the `ZeroShotClassificationPipeline`.
    ```python
    import torch
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

    pipeline = ZeroShotClassificationPipeline(
        model, # pretrained model
        tokenizer, # pretrained tokenizer
        classification_type='single-label', # problem type 
        # use `multi-label` for multi-label classification
        device=device
    )
    ```

2. **Classify a Text** 
    ```python
    text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US$394.3 billion in 2022 revenue. As of March 2023, Apple is the world's biggest company by market capitalization. As of June 2022, Apple is the fourth-largest personal computer vendor by unit sales and the second-largest mobile phone manufacturer in the world. It is considered one of the Big Five American information technology companies, alongside Alphabet (parent company of Google), Amazon, Meta Platforms, and Microsoft. 
    Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014.
    Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. Apple went public in 1980 to instant financial success. The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement called "1984". By 1985, the high cost of its products, and power struggles between executives, caused problems. Wozniak stepped back from Apple and pursued other ventures, while Jobs resigned and founded NeXT, taking some Apple employees with him. 
    """
    labels = ["business", "computers", "sport", "politics", "science"]
    results = pipeline(
        text,
        labels,
        threshold=0.5,
        batch_size=8
    )[0] # Select 1st as we only have one text

    for result in results:
        print(f"{result['label']} => {result['score']:.3f}")
    ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    business => 0.519
    ```
   </details>

### Zero-Shot Classification with labels chunking 

The `ZeroShotClassificationWithLabelsChunkingPipeline` is a pipeline for text classification that works almost identically to the `ZeroShotClassificationPipeline`, except that it pre-breaks the labels' list into chunks of a specified size.
:::tip
Use this pipeline if you want to classify a large number of labels (more than 25 per text)
:::

1. **Initialize the Pipeline**  

    ```python 
    from gliclass import ZeroShotClassificationWithLabelsChunkingPipeline

    pipeline = ZeroShotClassificationWithLabelsChunkingPipeline(
        model, # pretrained model
        tokenizer, # pretrained tokenizer
        classification_type='multi-labe', # problem type 
        # use `single-label` for single-label classification
        device=device
    )

2. **Classify a Text** 
    ```python
    text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US$394.3 billion in 2022 revenue. As of March 2023, Apple is the world's biggest company by market capitalization. As of June 2022, Apple is the fourth-largest personal computer vendor by unit sales and the second-largest mobile phone manufacturer in the world. It is considered one of the Big Five American information technology companies, alongside Alphabet (parent company of Google), Amazon, Meta Platforms, and Microsoft. 
    Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014.
    Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. Apple went public in 1980 to instant financial success. The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement called "1984". By 1985, the high cost of its products, and power struggles between executives, caused problems. Wozniak stepped back from Apple and pursued other ventures, while Jobs resigned and founded NeXT, taking some Apple employees with him. 
    """

    labels = [
    "business", "computers", "science", "innovation", "success", "leadership", "growth", "achievement", "entrepreneurship", "breakthrough", "market leader", "revolutionary", "profitable", "setbacks", "disputes", "crisis", "conflict", "problems", "struggles", "resignation", "high cost", "power struggles", "departure", "difficulties", "expensive", "competition", "rivalry", "challenges"
    ]

    results = pipeline(
        text,
        labels,
        threshold=0.85,
        labels_chunk_size = 15 # specify chunk size for your labels
    )[0]

    for result in results:
        print(f"{result['label']} => {result['score']:.3f}")

    ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    business => 1.000
    innovation => 1.000
    growth => 1.000
    entrepreneurship => 1.000
    leadership => 1.000
    success => 1.000
    achievement => 1.000
    breakthrough => 1.000
    computers => 1.000
    profitable => 1.000
    science => 1.000
    revolutionary => 1.000
    market leader => 1.000
    challenges => 1.000
    resignation => 1.000
    departure => 1.000
    expensive => 1.000
    high cost => 1.000
    difficulties => 0.999
    problems => 0.999
    setbacks => 0.998
    competition => 0.996
    rivalry => 0.995
    crisis => 0.992
    struggles => 0.991
    conflict => 0.987
    disputes => 0.949
    power struggles => 0.899
    ```
   </details>