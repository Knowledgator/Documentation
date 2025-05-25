# Components & Configs

## GLiNERConfig <sup><a href="https://github.com/urchade/GLiNER/blob/main/gliner/config.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>


The configuration class used to define the architecture and behavior of a GLiNER model. It inherits from [`PretrainedConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/configuration).

This class is used to control key architectural aspects, including the encoder, label encoder, span representation strategy, and additional fusion or RNN layers.

---
### Parameters
#### `model_name` 
`str`, *optional*, defaults to `"microsoft/deberta-v3-small"`

Base encoder model identifier from Hugging Face Hub or local path.

---

#### `labels_encoder`
`str`, *optional*

Encoder model to be used for embedding label texts. Can be a model ID or local path.

---

#### `name`
`str`, *optional*, defaults to `"span level gliner"`

Optional display name for this model configuration.

---

#### `max_width`
`int`, *optional*, defaults to `12`

Maximum span width (in number of tokens) allowed when generating candidate spans.

---

#### `hidden_size`
`int`, *optional*, defaults to `512`

Dimensionality of hidden representations in internal layers.

---

#### `dropout`
`float`, *optional*, defaults to `0.4`

Dropout rate applied to intermediate layers.

---

#### `fine_tune`
`bool`, *optional*, defaults to `True`

Whether to fine-tune the encoder during training.

---

#### `subtoken_pooling`
`str`, *optional*, defaults to `"first"`

Strategy used to pool subword token embeddings.  
**Choices:** `"first"`, `"mean"`, `"last"`

---

#### `span_mode` <sup><a href="https://github.com/urchade/GLiNER/blob/main/gliner/modeling/span_rep.py#L312" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`str`, *optional*, defaults to `"markerV0"`

Type: `str` — *optional*, defaults to `"markerV0"`  
Defines the strategy for constructing span representations from encoder outputs.

**Available options:**

- `"markerV0"` — Projects start and end token representations with MLPs and concatenates them. Lightweight and default.
- `"marker"` — Similar to `markerV0` but with deeper two-layer projections; better for complex tasks.
- `"query"` — Uses learned per-span-width query vectors and dot product interaction.
- `"mlp"` — Applies a feedforward MLP and reshapes output into span format; fast but position-agnostic.
- `"cat"` — Concatenates token features with learned span width embeddings before projection.
- `"conv_conv"` — Uses multiple 1D convolutions with increasing kernel sizes; captures internal structure.
- `"conv_max"` — Max pooling over tokens in span; emphasizes the strongest token.
- `"conv_mean"` — Mean pooling across span tokens.
- `"conv_sum"` — Sum pooling; raw additive representation.
- `"conv_share"` — Shared convolution kernel over span widths; parameter-efficient alternative.

  [**Read more**](#span-representation-layers)
---

#### `post_fusion_schema` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/layers.py#L126" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`str`, *optional*, defaults to `""`

Defines the multi-step attention schema used to fuse span and label embeddings. The value is a string with hyphen-separated tokens that determine the sequence of attention operations applied in the [`CrossFuser`](https://github.com/urchade/GLiNER/blob/main/gliner/modeling/fusion.py) module.

Each token in the schema defines one of the following attention types:

- `"l2l"` — **label-to-label self-attention** (intra-label interaction)
- `"t2t"` — **token-to-token self-attention** (intra-span interaction)
- `"l2t"` — **label-to-token cross-attention** (labels attend to span tokens)
- `"t2l"` — **token-to-label cross-attention** (tokens attend to labels)

**Examples:**

- `"l2l-l2t-t2t"` — apply label self-attention → label-to-token attention → token self-attention
- `"l2t"` — a single step where labels attend to span tokens
- `""` — disables fusion entirely (no interaction is applied)

The number and order of operations affects both performance and computational cost.

:::tip
The number of fusion layers (`num_post_fusion_layers`) controls how many times the entire schema is repeated.
:::
---

#### `num_post_fusion_layers`
`int`, *optional*, defaults to `1`

Number of layers applied after span-label fusion.

---

#### `vocab_size`
`int`, *optional*, defaults to `-1`

Vocabulary size override if needed (e.g. for decoder components).

---

#### `max_neg_type_ratio`
`int`, *optional*, defaults to `1`

Controls the ratio of negative (non-matching) types during training.

---

#### `max_types`
`int`, *optional*, defaults to `25`

Maximum number of entity types supported.

---

#### `max_len`
`int`, *optional*, defaults to `384`

Maximum sequence length accepted by the encoder.

---

#### `words_splitter_type`
`str`, *optional*, defaults to `"whitespace"`

Heuristic used for word-level splitting during inference.  

---

#### `has_rnn`
`bool`, *optional*, defaults to `True`

Whether to apply an LSTM on top of encoder outputs.

---

#### `fuse_layers`
`bool`, *optional*, defaults to `False`

If `True`, combine representations from multiple encoders (labels and main encoder).

---

#### `embed_ent_token`
`bool`, *optional*, defaults to `True`

If `True`, `<<ENT>>` tokens will be pooled for each label, if `False`, the first token of each label will be pooled as label embedding.

---

#### `class_token_index`
`int`, *optional*, defaults to `-1`

Index of the classification token in the encoder (e.g. `[CLS]`). Set `-1` if unused.

---

#### `encoder_config`
`dict` or `PretrainedConfig`, *optional*

A nested config dictionary for the encoder model. If a dict is passed, its `model_type` must be set or inferred.

---

#### `labels_encoder_config`
`dict` or `PretrainedConfig`, *optional*

Same as `encoder_config`, but used to configure the label encoder.

---

#### `ent_token`
`str`, *optional*, defaults to `"<<ENT>>"`

Token used to mark entity span boundaries.

---

#### `sep_token`
`str`, *optional*, defaults to `"<<SEP>>"`

Token used to separate entities or fields in span input.

---

#### `_attn_implementation`
`any`, *optional*

Optional override for attention logic (used in advanced configurations or experimentation).

**Examples:**

- Could be used to turn off flash-attention if `flashdeberta` or `flash-attn` is installed and model supports `Flash Attention`.
```python
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1", _attn_implementation="eager")
```

---

### Examples

#### Initiate model from config
```python
from gliner import GLiNERConfig, GLiNER

config = GLiNERConfig.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
model = GLiNER(config)
```

## TrainingArguments <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/training/trainer.py#L23" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Custom extension of `transformers.TrainingArguments` with additional parameters for span-based models, focal loss control, and parameter-specific optimization.

---

### Parameters

---

#### `cache_dir`  
`str`, *optional*  
Directory to store cache files.

---

#### `optim`  
`str`, *optional*, defaults to `"adamw_torch"`  
Optimizer name used during training.

---

#### `others_lr`  
`float`, *optional*  
Overrides learning rate for non-encoder parameters (e.g. label encoder, `token_rep_layer`). Used to create separate optimizer groups in `create_optimizer`.

---

#### `others_weight_decay`  
`float`, *optional*, defaults to `0.0`  
Weight decay used for non-encoder parameters. Only applied if `others_lr` is specified.

---

#### `focal_loss_alpha`  
`float`, *optional*, defaults to `-1`  
Alpha for focal loss. If ≥ 0, focal loss is activated.

Focal loss formula:  
FL(pₜ) = -α × (1 - pₜ)^γ × log(pₜ)

---

#### `focal_loss_gamma`  
`float`, *optional*, defaults to `0`  
Gamma for focal loss. Amplifies effect of hard examples.

---

#### `label_smoothing`  
`float`, *optional*, defaults to `0.0`  
Smoothing factor ε for regularizing classification targets.

Smoothed label formula:  
yᵢ_smooth = (1 - ε) × yᵢ + ε / N  
Where:  
- ε is the label smoothing factor  
- N is the number of classes

---

#### `loss_reduction`  
`str`, *optional*, defaults to `"sum"`  
Specifies how loss is aggregated.  
Choices: `"sum"`, `"mean"`

---

#### `negatives`  
`float`, *optional*, defaults to `1.0`  
Ratio of negative to positive spans during training. Controls sampling balance.

---

#### `masking`  
`str`, *optional*, defaults to `"global"`  
Controls masking strategy for spans.  
Choices:  
- `"global"` — fixed mask  
- `"softmax"` — attention-based masking  
- `"none"` — no masking

---


## Span Representation Layers

GLiNER supports multiple span representation strategies that define how text spans (e.g., entity candidates) are encoded using the contextualized token embeddings from the encoder. These are selected via the [`span_mode`](#span_mode-source) parameter in `GLiNERConfig`.

### Modes
---

#### `markerV0` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L262" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Projects the start and end token embeddings with MLPs, concatenates them, then applies a final projection. Lightweight and effective.

**Module:** `SpanMarkerV0`  
**Recommended use:** Default general-purpose span representation.

---

#### `marker` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L216" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Similar to `markerV0`, but uses deeper two-layer projections for start and end positions separately before fusion. More expressive but slightly more computationally expensive.

**Module:** `SpanMarker`  
**Recommended use:** When higher span-level abstraction is needed.

---

#### `query` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L7" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Uses learned per-width query vectors to extract span representations via a dot-product attention-like `einsum`. Resulting tensor is projected.

**Module:** `SpanQuery`  
**Recommended use:** When fixed span queries per width are desirable.

---

#### `mlp` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L33" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Applies a feedforward MLP over token embeddings and reshapes output to `[B, L, max_width, D]`.

**Module:** `SpanMLP`  
**Recommended use:** Efficient and parallel but ignores positional structure.

---

#### `cat` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L53" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Concatenates token embeddings with learned span-width embeddings, then projects the result. Explicitly models span width.

**Module:** `SpanCAT`  
**Recommended use:** When span length is a relevant feature.

---

#### `conv_conv` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L84" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Applies a series of 1D convolutions with increasing kernel sizes. Captures internal structure within spans.

**Module:** `SpanConv`  
**Recommended use:** Tasks that benefit from compositional features inside spans.

---

#### `conv_max` / `conv_mean` / `conv_sum` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L84" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Apply pooling (max, mean, sum) over spans using increasing kernel sizes.  
Behavior differs only by pooling type.

**Module:** `SpanConvBlock`  
**Recommended use:** When a fixed summary of span tokens is appropriate.

---

#### `conv_share` <sup><a href="https://github.com/urchade/GLiNER/blob/efbfa38211136657895372d33d4ee2fe11b6f11b/gliner/modeling/span_rep.py#L170" target="_blank" rel="noopener noreferrer">[source]</a></sup>

Applies a shared convolutional kernel with increasing receptive fields. Parameter-efficient.

**Module:** `ConvShare`  
**Recommended use:** When model size or shared pattern extraction is prioritized.

---

Each representation module returns a tensor of shape `[B, L, max_width, D]` and can be initiated interchangeably via the `SpanRepLayer` interface.

