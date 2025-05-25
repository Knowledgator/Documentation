# Usage

## 🚀 Basic Use Case

After the installation of the GLiNER library, import the `GLiNER` class. Following this, you can load your chosen model with `GLiNER.from_pretrained` and utilize `predict_entities` to discern entities within your text.

```python
from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

# Labels for entity prediction
labels = ["Person", "Award", "Date", "Competitions", "Teams"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

<details>
    <summary>Expected Output</summary>
    ```bash
    Cristiano Ronaldo dos Santos Aveiro => Person
    5 February 1985 => Date
    Al Nassr => Teams
    Portugal national team => Teams
    UEFA Men's Player of the Year Awards => Award
    European Golden Shoes => Award
    UEFA Champions Leagues => Competitions
    UEFA European Championship => Competitions
    UEFA Nations League => Competitions
    ```
</details>

## 🏃‍♀️Using FlashDeBERTa

Most GLiNER models use the DeBERTa encoder as their backbone. This architecture offers strong token classification performance and typically requires less data to achieve good results. However, a major drawback has been its slower inference speed, and until recently, there was no flash attention implementation compatible with DeBERTa's disentangled attention mechanism.

To address this, [FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa) was introduced.

To use `FlashDeBERTa` with GLiNER, install it with:

```bash
pip install flashdeberta -U
```
:::tip
Before using FlashDeBERTa, please make sure that you have `transformers>=4.47.0`.
:::

GLiNER will automatically detect and use `FlashDeBERTa`. If needed, you can switch to the standard `eager` attention mechanism by specifying the attention implementation:

```python
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1", _attn_implementation="eager")
```

`FlashDeBERTa` provides up to a 3× speed boost for typical sequence lengths—and even greater improvements for longer sequences.


## 🛠️Pipelines {#pipelines}
GLiNER-Multitask models are designed to extract relevant information from plain text based on a user-provided custom prompt. The advantage of such encoder-based multitask models is that they enable efficient and more controllable information extraction with a single model that reduces costs on computational and storage resources. Moreover, such encoder models are more interpretable, efficient and tunable than LLMs, which are hard to fine-tune and use for information extraction.

**Supported tasks:**:
   * Named Entity Recognition (NER): Identifies and categorizes entities such as names, organizations, dates, and other specific items in the text.
   * Relation Extraction: Detects and classifies relationships between entities within the text.
   * Summarization: Extract the most important sentences that summarize the input text, capturing the essential information.
   * Sentiment Extraction: Identify parts of the text that signalize a positive, negative, or neutral sentiment;
   * Key-Phrase Extraction: Identifies and extracts important phrases and keywords from the text.
   * Question-answering: Finding an answer in the text given a question;
   * Open Information Extraction: Extracts pieces of text given an open prompt from a user, for example, product description extraction;
   * Text classification: Classifying text by matching labels specified in the prompt;

We prepared high-level classes that simplify the usage and evaluation of GLiNER multi-task models for different task types.

### Classification

The `GLiNERClassifier` is a pipeline for text classification tasks based on the GLiNER model. It evaluates input text against a set of predefined labels, supporting both single-label and multi-label classification. It also calculates F1 scores for evaluation on datasets.

1. **Initialize the Classifier**  
   Load a pretrained model and initialize the `GLiNERClassifier`.

   ```python
   from gliner import GLiNER
   from gliner.multitask import GLiNERClassifier

   model_id = 'knowledgator/gliner-multitask-v1.0'
   model = GLiNER.from_pretrained(model_id)
   classifier = GLiNERClassifier(model=model)
   ```

2. **Classify a Text**  
   Classify a single text into a list of labels.

   ```python
   text = "SpaceX successfully launched a new rocket into orbit."
   labels = ['science', 'technology', 'business', 'sports']
   predictions = classifier(text, classes=labels, multi_label=False)
   print(predictions)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    [[{'label': 'technology', 'score': 0.3839840292930603}]]
    ```
   </details>

3. **Evaluate on a Dataset**  
   Evaluate the model on a dataset from Hugging Face.

   ```python
   metrics = classifier.evaluate('dair-ai/emotion')
   print(metrics)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    {'micro': 0.4465, 'macro': 0.42431600571236294, 'weighted': 0.48842385572766434}
    ```
   </details>
### Question-Answering

The `GLiNERQuestionAnswerer` is a pipeline for question-answering tasks based on the GLiNER model. It extracts answers based on questions and input text. You can leverage `GLiNERSquadEvaluator` to evaluate a model on the SQuAD dataset.

1. **Initialize the Question-Answerer**  
   Load a pretrained model and initialize the `GLiNERQuestionAnswerer`.

   ```python
   from gliner import GLiNER
   from gliner.multitask import GLiNERQuestionAnswerer

   model_id = 'knowledgator/gliner-multitask-v1.0'
   model = GLiNER.from_pretrained(model_id)
   answerer = GLiNERQuestionAnswerer(model=model)
   ```
2. **Extract an answer from a Text**  
   Extract an answer to the input question.

   ```python
   text = "SpaceX successfully launched a new rocket into orbit."
   question = 'Which company launched a new rocker?'
   predictions = answerer(text, questions=question)
   print(predictions)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    [[{'answer': 'SpaceX', 'score': 0.998126208782196}]]
    ```
   </details>


3. **Evaluate on a Dataset**  
   Evaluate the model on a dataset from Hugging Face.

   ```python
   from gliner.multitask import GLiNERSquadEvaluator
   model_id = 'knowledgator/gliner-multitask-v1.0'
   evaluator = GLiNERSquadEvaluator(model_id=model_id)
   metrics = evaluator.evaluate(threshold=0.25)
   print(metrics)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    {'exact': 29.411269266402762, 'f1': 29.80174667785213, 'total': 11873, 'HasAns_exact': 0.6916329284750338, 'HasAns_f1': 1.473707541521307, 'HasAns_total': 5928, 'NoAns_exact': 58.048780487804876, 'NoAns_f1': 58.048780487804876, 'NoAns_total': 5945, 'best_exact': 50.08001347595385, 'best_exact_thresh': 0.0, 'best_f1': 50.08001347595385, 'best_f1_thresh': 0.0}
    ```
   </details>


### Relation Extraction

The `GLiNERRelationExtractor` is a pipeline for extracting relationships between entities in a text using the GLiNER model. The pipeline combines both zero-shot named entity recognition and relation extraction. It identifies entity pairs and their relations based on a specified by user set of relation types.
S
1. **Initialize the Relation Extractor**  
   Load a pretrained model and initialize the `GLiNERRelationExtractor`.

   ```python
   from gliner import GLiNER
   from gliner.multitask import GLiNERRelationExtractor

   model_id = 'knowledgator/gliner-multitask-v1.0'
   model = GLiNER.from_pretrained(model_id)
   relation_extractor = GLiNERRelationExtractor(model=model)
   ```

2. **Extract Relations from Text**  
   Identify relationships between entities in a given text.

   ```python
   text = "Elon Musk founded SpaceX in 2002 to reduce space transportation costs."
   relations = ['founded', 'owns', 'works for']
   entities = ['person', 'company', 'year']
   predictions = relation_extractor(text, entities=entities, relations=relations)
   print(predictions)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    [[{'source': 'Elon Musk', 'relation': 'founded', 'target': 'SpaceX', 'score': 0.9583475589752197}]]
    ```
   </details>


For more nuance tuning of relation extraction pipeline, we recommend to use `utca` framework.

### Open Information Extraction

The `GLiNEROpenExtractor` is a pipeline designed to extract information from a text given a user query. By default in terms of GLiNER labels `match` tag is used, however, we recommend combining prompting and selecting appropriate tags for your tasks. 

1. **Initialize the Information Extractor**  
   Load a pretrained model and initialize the `GLiNEROpenExtractor`.

   ```python
   from gliner import GLiNER
   from gliner.multitask import GLiNEROpenExtractor

   model_id = 'knowledgator/gliner-multitask-v1.0'
   model = GLiNER.from_pretrained(model_id)
   extractor = GLiNEROpenExtractor(model=model, prompt="Extract all companies related to space technologies")
   ```

2. **Extract Information from Text**  
   Identify relevant information from a given text.

   ```python
   text = "Elon Musk founded SpaceX in 2002 to reduce space transportation costs. Also Elon is founder of Tesla, NeuroLink and many other companies."
   labels = ['company']
   predictions = extractor(text, labels=labels)
   print(predictions)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    [[{'start': 72, 'end': 78, 'text': 'SpaceX', 'label': 'company', 'score': 0.9622299075126648}, {'start': 149, 'end': 154, 'text': 'Tesla', 'label': 'company', 'score': 0.9357912540435791}, {'start': 156, 'end': 165, 'text': 'NeuroLink', 'label': 'company', 'score': 0.9122058749198914}]]
    ```
   </details>


### Summarization

The `GLiNERSummarizer` pipeline leverages the GLiNER model for performing summarization tasks as extraction process. 


1. **Initialize the Summarizer**  
   Load a pretrained model and initialize the `GLiNERSummarizer`.

   ```python
   from gliner import GLiNER
   from gliner.multitask import GLiNERSummarizer

   model_id = 'knowledgator/gliner-multitask-v1.0'
   model = GLiNER.from_pretrained(model_id)
   summarizer = GLiNERSummarizer(model=model)
   ```

2. **Summarize the Text**  
   Extract the most important information from a given text and construct summary.

   ```python
   text = "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014."
   summary = summarizer(text, threshold=0.1)
   print(summary)
   ```
   <details>
    <summary>Expected Output</summary>
    ```bash
    ['Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014.']

    ```
   </details>


## Relations extraction pipeline with [utca](https://github.com/Knowledgator/utca)
1. **Install utca**
:::danger Important
First of all, we need import neccessary components of the library and initialize predictor - GLiNER model and construct pipeline that combines NER and realtions extraction:
:::
```bash
pip install -U utca
```
2. **Initialize the Pipeline** 
```python
from utca.core import RenameAttribute
from utca.implementation.predictors import (
    GLiNERPredictor,
    GLiNERPredictorConfig
)
from utca.implementation.tasks import (
    GLiNER,
    GLiNERPreprocessor,
    GLiNERRelationExtraction,
    GLiNERRelationExtractionPreprocessor,
)

predictor = GLiNERPredictor( # Predictor manages the model that will be used by tasks
    GLiNERPredictorConfig(
        model_name = "knowledgator/gliner-multitask-v1.0", # Model to use
        device = "cuda:0", # Device to use
    )
)

pipe = (
    GLiNER( # GLiNER task produces classified entities that will be at the "output" key.
        predictor=predictor,
        preprocess=GLiNERPreprocessor(threshold=0.7) # Entities threshold
    ) 
    | RenameAttribute("output", "entities") # Rename output entities from GLiNER task to use them as inputs in GLiNERRelationExtraction
    | GLiNERRelationExtraction( # GLiNERRelationExtraction is used for relation extraction.
        predictor=predictor,
        preprocess=(
            GLiNERPreprocessor(threshold=0.5) # Relations threshold
            | GLiNERRelationExtractionPreprocessor()
        )
    )
)
```

3. **Run the pipeline**

To run pipeline we need to specify entity types and relations with their parameters:

```python
text = "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014."

r = pipe.run({
    "text": text, # Text to process
    "labels": ["organisation", "founder", "position", "date"],
    "relations": [{ # Relation parameters
        "relation": "founder", # Relation label. Required parameter.
        "pairs_filter": [("organisation", "founder")], # Optional parameter. It specifies possible members of relations by their entity labels.
        "distance_threshold": 100, # Optional parameter. It specifies the max distance between spans in the text (i.e., the end of the span that is closer to the start of the text and the start of the next one).
    }, {
        "relation": "inception date",
        "pairs_filter": [("organisation", "date")],
    }, {
        "relation": "held position",
        "pairs_filter": [("founder", "position")],
    }]
})

print(r["output"])
```

<details>
<summary>Expected Output</summary>
```bash
[{'source': {'start': 0, 'end': 9, 'span': 'Microsoft', 'score': 0.997672975063324, 'entity': 'organisation'}, 'relation': 'founder', 'target': {'start': 25, 'end': 35, 'span': 'Bill Gates', 'score': 0.9939975738525391, 'entity': 'founder'}, 'score': 0.9683231711387634}, {'source': {'start': 0, 'end': 9, 'span': 'Microsoft', 'score': 0.997672975063324, 'entity': 'organisation'}, 'relation': 'founder', 'target': {'start': 40, 'end': 50, 'span': 'Paul Allen', 'score': 0.9759964942932129, 'entity': 'founder'}, 'score': 0.8631041049957275}, {'source': {'start': 149, 'end': 158, 'span': 'Microsoft', 'score': 0.9986124038696289, 'entity': 'organisation'}, 'relation': 'founder', 'target': {'start': 40, 'end': 50, 'span': 'Paul Allen', 'score': 0.9759964942932129, 'entity': 'founder'}, 'score': 0.8631041049957275}, {'source': {'start': 0, 'end': 9, 'span': 'Microsoft', 'score': 0.997672975063324, 'entity': 'organisation'}, 'relation': 'inception date', 'target': {'start': 54, 'end': 67, 'span': 'April 4, 1975', 'score': 0.9899335503578186, 'entity': 'date'}, 'score': 0.9971309304237366}, {'source': {'start': 149, 'end': 158, 'span': 'Microsoft', 'score': 0.9986124038696289, 'entity': 'organisation'}, 'relation': 'inception date', 'target': {'start': 54, 'end': 67, 'span': 'April 4, 1975', 'score': 0.9899335503578186, 'entity': 'date'}, 'score': 0.9971309304237366}, {'source': {'start': 25, 'end': 35, 'span': 'Bill Gates', 'score': 0.9939975738525391, 'entity': 'founder'}, 'relation': 'held position', 'target': {'start': 188, 'end': 196, 'span': 'chairman', 'score': 0.9755746126174927, 'entity': 'position'}, 'score': 0.9663875102996826}, {'source': {'start': 25, 'end': 35, 'span': 'Bill Gates', 'score': 0.9939975738525391, 'entity': 'founder'}, 'relation': 'held position', 'target': {'start': 198, 'end': 221, 'span': 'chief executive officer', 'score': 0.9835125803947449, 'entity': 'position'}, 'score': 0.9466949105262756}, {'source': {'start': 25, 'end': 35, 'span': 'Bill Gates', 'score': 0.9939975738525391, 'entity': 'founder'}, 'relation': 'held position', 'target': {'start': 223, 'end': 232, 'span': 'president', 'score': 0.9777029156684875, 'entity': 'position'}, 'score': 0.9731078147888184}, {'source': {'start': 25, 'end': 35, 'span': 'Bill Gates', 'score': 0.9939975738525391, 'entity': 'founder'}, 'relation': 'held position', 'target': {'start': 237, 'end': 261, 'span': 'chief software architect', 'score': 0.9814156293869019, 'entity': 'position'}, 'score': 0.9499841332435608}]
```
</details>
