# Knowledgator Datasets Hub

Knowledgator offers a curated collection of high-quality datasets tailored for training and evaluating models across a wide range of **information extraction** tasks. These datasets are optimized for real-world scenarios, support multi-lingual and domain-specific use cases, and are compatible with Knowledgator frameworks like GLiNER, GLiClass, and LiqFit.

---

## Dataset Categories

### General Purpose  
Versatile datasets designed for broad NLP tasks such as text classification, relation extraction, and entity recognition.

- `knowledgator/GLINER-multi-task-synthetic-data`: Synthetic multi-task dataset for NER, RE, summarization, sentiment, QA, and more.
- `gliclass-v2.0` / `gliclass-v2.0-RAC`: Text classification dataset with optional retrieval-augmented context.

---

### Multilingual  
Cross-lingual datasets supporting named entity recognition in 10+ languages.

- `tner/multinerd`: Multilingual, multi-genre NER annotations.
- `MultiCoNER/multiconer_v2`: Handles low-context and complex named entities across 12 languages.
- `unimelb-nlp/wikiann`: WikiANN dataset for 176 languages.

---

### Biomedical  
Specialized corpora for biomedical and chemical domain tasks such as disease recognition, gene tagging, and adverse effect extraction.

- `bigbio/anat_em`, `drugprot`, `chia`, `nlm_gene`, `scai_disease`, and more: Cover anatomical, pharmaceutical, and clinical trial data.

---

## Highlights

- Supports **NER**, **RE**, **Text Classification**, **Keyphrase Extraction**, **QA**, and more.
- Datasets annotated manually, synthetically, or with weak supervision.
- Available via Hugging Face Datasets or directly through Knowledgator tools.
- Structured to work with few-shot and retrieval-augmented learning.