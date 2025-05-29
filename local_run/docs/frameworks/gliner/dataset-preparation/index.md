# GLiNER Datasets Documentation

*Guide for GLiNER datasets: Format, preparation, and overview of already prepared data*

---

## Dataset Format Specification

GLiNER models require training data in a structured JSON format for optimal performance:

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

## Quick Start Guide

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

---

## Prepared Dataset Collection

Ready-to-use datasets optimized for GLiNER training across different domains and languages.

### General Purpose Datasets

#### knowledgator/GLINER-multi-task-synthetic-data
[Dataset Link](https://huggingface.co/datasets/knowledgator/GLINER-multi-task-synthetic-data)

The flagship synthetic dataset powering GLiNER's multi-task capabilities. Engineered to handle diverse NLP challenges in a unified framework.

**Supported Tasks:**
- **Named Entity Recognition**: Extract and categorize entities (names, organizations, dates)
- **Relation Extraction**: Identify relationships between text entities
- **Text Summarization**: Extract key sentences capturing essential information
- **Sentiment Analysis**: Detect positive, negative, or neutral sentiment regions
- **Key-Phrase Extraction**: Identify important phrases and keywords
- **Question Answering**: Locate answers within text given questions
- **Open Information Extraction**: Extract content based on custom user prompts

```json
{
  "total_examples": 48548,
  "unique_entities": 50094
}
```

#### tner/tweebank_ner
[Dataset Link](https://huggingface.co/datasets/tner/tweebank_ner)

Social media NER dataset from the TNER project, optimized for Twitter-style text processing with informal language patterns.

```json
{
  "total_examples": 1639,
  "unique_entities": 4,
  "entities": ["location", "organization", "other", "person"]
}
```

#### thunlp/docred
[Dataset Link](https://huggingface.co/datasets/thunlp/docred)

Document-level relation extraction dataset built from Wikipedia and Wikidata. Designed for complex inter-sentence relationship understanding across entire documents.

**Key Features:**
- Multi-sentence context processing
- Complex entity relationship inference
- Large-scale distantly supervised data available

```json
{
  "total_examples": 3053,
  "unique_entities": 6,
  "entities": [
    "location", "numerical entity", "organization", 
    "other", "person", "time"
  ]
}
```

---

### Multilingual Dataset Collection

Cross-lingual datasets supporting diverse languages and cultural contexts.

#### tner/multinerd
[Dataset Link](https://huggingface.co/datasets/tner/multinerd)

First language-agnostic methodology for creating multilingual, multi-genre NER annotations. Covers 10 languages with fine-grained entity categories.

**Languages:** Chinese, Dutch, English, French, German, Italian, Polish, Portuguese, Russian, Spanish  
**Genres:** Wikipedia articles, WikiNews content

```json
{
  "total_examples": 2283360,
  "unique_entities": 15,
  "entities": [
    "animal", "biological entity", "celestial body", "disease", 
    "event", "food", "institution", "location", "media", 
    "mythological entity", "organization", "person", "plant", 
    "time", "vehicle"
  ]
}
```

#### MultiCoNER/multiconer_v2
[Dataset Link](MultiCoNER/multiconer_v2)

Large-scale multilingual NER dataset addressing contemporary challenges including low-context scenarios and syntactically complex entities.

**Languages:** Bangla, Chinese, English, Spanish, Farsi, French, German, Hindi, Italian, Portuguese, Swedish, Ukrainian  
**Domains:** Wiki sentences, questions, search queries

```json
{
  "total_examples": 154046,
  "unique_entities": 33,
  "entities": [
    "aerospace manufacturer", "anatomical structure", "art work", 
    "athlete", "car manufacturer", "cleric", "clothing", "disease", 
    "drink", "facility", "food", "human settlement", "location", 
    "medical procedure", "medication/vaccine", "musical group", 
    "organization", "person", "politician", "product", "software", 
    "sports group", "vehicle", "visual work", "written work"
  ]
}
```

#### Synthetic Multilingual Dataset
[Dataset Link](#)

AI-generated multilingual dataset created using Qwen3-4B model annotations on [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) content.

```json
{
  "total_examples": 41197,
  "unique_entities": 27123
}
```

#### unimelb-nlp/wikiann
[Dataset Link](https://huggingface.co/datasets/unimelb-nlp/wikiann)

WikiANN (PAN-X) multilingual NER dataset spanning 176 languages with balanced train/dev/test splits.

```json
{
  "total_examples": 678900,
  "unique_entities": 3,
  "entities": ["location", "organization", "person"]
}
```

---

### Biomedical Dataset Collection

Specialized datasets for medical, biological, and chemical entity recognition.

#### bigbio/anat_em
[Dataset Link](https://huggingface.co/datasets/bigbio/anat_em)

Extended Anatomical Entity Mention corpus with 250K+ words manually annotated for anatomical entities using 12 granularity-based categories.

```json
{
  "total_examples": 606,
  "unique_entities": 12,
  "entities": [
    "anatomical system", "cancer", "cell", "cellular component",
    "developing anatomical structure", "immaterial anatomical entity",
    "multi", "organ", "organism subdivision", "organism substance",
    "pathological formation", "tissue"
  ]
}
```

#### bigbio/drugprot
[Dataset Link](https://huggingface.co/datasets/bigbio/anat_em)

DrugProt corpus featuring expert-labeled chemical and gene mentions with biologically relevant relationships from BioCreative VII.

```json
{
  "total_examples": 3500,
  "unique_entities": 3,
  "entities": ["chemical", "gene", "other"]
}
```

#### bigbio/muchmore
[Dataset Link](https://huggingface.co/datasets/bigbio/muchmore)

Parallel English-German medical abstracts corpus from 41 medical journals, each representing distinct medical sub-domains.

```json
{
  "total_examples": 7822,
  "unique_entities": 2,
  "entities": ["other", "umlsterm"]
}
```

#### bigbio/chia
[Dataset Link](https://huggingface.co/datasets/bigbio/chia)

Large annotated corpus of patient eligibility criteria from 1,000 Phase IV clinical trials with comprehensive entity and relationship annotations.

```json
{
  "total_examples": 2000,
  "unique_entities": 17,
  "entities": [
    "condition", "device", "drug", "measurement", "mood",
    "multiplier", "negation", "observation", "other", "person",
    "procedure", "qualifier", "reference point", "scope",
    "temporal", "value", "visit"
  ]
}
```

#### bigbio/bioinfer
[Dataset Link](https://huggingface.co/datasets/bigbio/bioinfer)

Protein, gene, and RNA relationship corpus with 1,100 sentences annotated for relationships, entities, and syntactic dependencies.

```json
{
  "total_examples": 894,
  "unique_entities": 5,
  "entities": [
    "gene", "individual protein", "other",
    "protein complex", "protein family or group"
  ]
}
```

#### bigbio/scai_disease
[Dataset Link](#https://huggingface.co/datasets/bigbio/scai_disease)

Disease and adverse effects dataset with 400 MEDLINE abstracts annotated by life sciences Masters degree holders.

```json
{
  "total_examples": 400,
  "unique_entities": 2,
  "entities": ["adverse", "disease"]
}
```

#### bigbio/nlm_gene
[Dataset Link](https://huggingface.co/datasets/bigbio/nlm_gene)

Comprehensive gene recognition corpus with 550 PubMed articles covering 28 organisms and 15K+ unique gene names.

```json
{
  "total_examples": 450,
  "unique_entities": 3,
  "entities": ["gene", "gene or gene product", "other"]
}
```

#### bigbio/seth_corpus
[Dataset Link](https://huggingface.co/datasets/bigbio/seth_corpus)

SNP (Single Nucleotide Polymorphism) named entity recognition corpus from 630 PubMed citations.

```json
{
  "total_examples": 630,
  "unique_entities": 4,
  "entities": ["gene", "other", "rs", "snp"]
}
```

#### bigbio/ctebmsp
[Dataset Link](https://huggingface.co/datasets/bigbio/ctebmsp)

Spanish clinical trials corpus with 500 Creative Commons licensed abstracts from evidence-based medicine studies.

```json
{
  "total_examples": 300,
  "unique_entities": 5,
  "entities": ["anatomy", "chemical", "disorder", "other", "process"]
}
```

#### bigbio/mirna
[Dataset Link](bigbio/mirna)

MicroRNA corpus with 301 Medline citations manually annotated for gene, disease, and miRNA entities.

```json
{
  "total_examples": 201,
  "unique_entities": 7,
  "entities": [
    "diseases", "genes", "non", "other",
    "relation trigger", "species", "specific mirnas"
  ]
}
```

#### bigbio/gnormplus
[Dataset Link](https://huggingface.co/datasets/bigbio/gnormplus)

Enhanced gene corpus combining BioCreative II GN and Citation GIA datasets with added gene family and protein domain annotations.

```json
{
  "total_examples": 432,
  "unique_entities": 4,
  "entities": ["Gene", "domain motif", "family name", "other"]
}
```

#### bigbio/osiris
[Dataset Link](https://huggingface.co/datasets/bigbio/gnormplus) 

MEDLINE abstracts manually annotated for human genetic variation mentions under Creative Commons licensing.

```json
{
  "total_examples": 105,
  "unique_entities": 2,
  "entities": ["gene", "variant"]
}
```