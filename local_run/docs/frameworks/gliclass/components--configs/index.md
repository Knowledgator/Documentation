# Components & Configs

## GLiClassModelConfig <sup><a href="https://github.com/Knowledgator/GLiClass/blob/19bf6a124a8fe9a81720591cf9980c5870738a82/gliclass/config.py#L8C7-L8C26" target="_blank" rel="noopener noreferrer">[source]</a></sup>


The configuration class used to define the architecture and behavior of a GLiClass model. It inherits from [`PretrainedConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/configuration).

This class is used to control key architectural aspects, including the encoder, label encoder, projection strategy, scorers or LSTM layers.

---
### Parameters
#### `encoder_config`
`dict|None`, *optional*

Encoder model configuration

---
#### `encoder_model`
`str|None`, *optional*

Base encoder model identifier from Hugging Face Hub or local path.

---
#### `label_model_config`
`dict|None`, *optional*

Label encoder model configuration
:::note
Requiered for `bi-encoder` architecture
:::

---
#### `label_model_name`
`str|None`, *optional*

Encoder model to be used for embedding label texts. Can be a model ID or local path.
:::note
Requiered for `bi-encoder` architecture
:::

---
#### `class_token_index`
`int`, *optional*, defaults to `-1`

Сlass token index

---
#### `text_token_index`
`int`, *optional*, defaults to `-1`

Text token index

---
#### `ignore_index`
`int`, *optional*, defaults to `-100`

Index to ignore when calculating loss

---
#### `hidden_size`
`int|None`, *optional*

Dimensionality of hidden representations in internal layers.

---
#### `projector_hidden_act`
`str`, *optional*, defaults to `gelu`

Projector activation function, for avaliable options, please see [transformers repository](https://github.com/huggingface/transformers/blob/e8b292e35f331d3c3de85f7e5d3496b0e13d3d6f/src/transformers/activations_tf.py#L127).

---
#### `vocab_size`
`int|None`, *optional*

Dictionary size (taken from encoder_config)

---
#### `problem_type`
`str|None`, *optional*, defaults to `single_label_classification`

Defines the type of classification problem and determines the loss function used during training.

**Available options**:  

- `"regression"` — Regression task for predicting continuous values. Uses MSELoss for single or multi-output regression.
- `"single_label_classification"` — Standard multi-class classification where each sample belongs to exactly one class. Uses CrossEntropyLoss. Default option.
- `"multi_label_classification"` — Multi-label classification where each sample can belong to multiple classes simultaneously. Uses focal loss with configurable alpha and gamma parameters.
- `None` - Automatic problem type detection based on label structure: regression for single label, CrossEntropyLoss for 1D labels, and LogSoftmax for multi-dimensional soft labels.

---
#### `max_num_classes`
`Any`, *optional*, defaults to `25`

Description for `max_num_classes`.

---
#### `use_lstm`
`bool`, *optional*, defaults to `False`

Flag to choose LSTM usage in GLiClass

---
#### `initializer_range`
`float`, *optional*, defaults to `0.03`

Weights initialization range

---
#### `scorer_type`
`str`, *optional*, defaults to `simple`

Defines the scoring mechanism used to compute similarity between text representations and class label representations.  

**Available options:**

- `"weighted-dot"` - Projects both text and labels into 2x hidden size, creates two representation pairs, concatenates them with element-wise product, and applies MLP.
- `"simple"` - Computes direct dot product between text and label embeddings using einsum. Fastest and most straightforward approach, suitable for most tasks.
- `"mlp"` - Concatenates text and label representations, then passes through 3-layer MLP (2×hidden_size → mlp_hidden_size → mlp_hidden_size/2 → 1).
- `"hopfield"` - Uses Hopfield attention mechanism with Q/K/V projections and iterative refinement of label representations.

---
#### `pooling_strategy`
`str`, *optional*, defaults to `first`

Defines the pooling strategy used to aggregate token-level representations into a single sequence representation.

**Available options:**  
- `"max"` -  Applies global max pooling across the sequence dimension.
- `"first"` - Takes the first token's embedding.
- `"last"` -  Takes the last token's embedding. 
- `"avg"` - Applies global average pooling with optional attention mask support.
- `"sum"` -  Applies global sum pooling with optional attention mask support.
- `"rms"` - Applies Root Mean Square pooling with attention mask support.
- `"abs_max"` -  Applies max pooling on absolute values with attention mask support.
- `"abs_avg"` - Applies average pooling on absolute values with attention mask support.

---
#### `focal_loss_alpha`
`float`, *optional*, defaults to `0.5`

Alpha parameter for the focal loss.

---
#### `focal_loss_gamma`
`float`, *optional*, defaults to `2`

Gamma parameter for the focal loss.

---
#### `logit_scale_init_value`
`float`, *optional*, defaults to `2.6592`

Initial value of logit scale

---
#### `normalize_features`
`bool`, *optional*, defaults to `False`

Flag for normalizing features

---
#### `extract_text_features`
`bool`, *optional*, defaults to `False`

Flag for extracting text features

---
#### `contrastive_loss_coef`
`float`, *optional*, defaults to `0`

Contrastive loss coefficient

---
#### `architecture_type`
`str`, *optional*, defaults to `uni-encoder`

Defines the architectural approach used for encoding text and class representations in the GLiClass model.  

**Available options:**    
- `"uni-encoder"` — Single encoder processes both text and class tokens in the same sequence. Classes are embedded as special tokens within the input.
- `"bi-encoder"` — Separate encoders for text and class labels. Text encoder processes input text, while label encoder processes class descriptions independently. Allows for different model architectures for each component.
- `"bi-encoder-fused"` — Extension of bi-encoder where class embeddings from the label encoder are fused back into the text encoder at class token positions.
- `"encoder-decoder"` — Uses encoder-decoder architecture where text is processed by the encoder and class information by the decoder. Requires models with `is_encoder_decoder=True` configuration.
---
#### `prompt_first`
`bool`, *optional*, defaults to `False`

Flag to choose whether to place labels at the beginning

---
#### `squeeze_layers`
`bool`, *optional*, defaults to `False`

Flag to choose whether to compress layers

---
#### `embed_class_token`
`Any`, *optional*, defaults to `True`

If `True`, `<<LABEL>>` tokens will be pooled for each label, if `False`, the first token of each label will be pooled as label embedding.

---


