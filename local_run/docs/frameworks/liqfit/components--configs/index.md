# Components & Configs

## Collators <sup><a href="https://github.com/Knowledgator/LiqFit/tree/main/src/liqfit/collators" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---
### NLICollator <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/collators/nli_collator.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.collators.NLICollator`
#### Parameters
##### `tokenizer`  
`AutoTokenizer`, *Callable*  

The tokenizer used to process input text into input IDs.  

---

##### `max_length`  
*int*  

Maximum length applied during tokenization of input sequences.  

---

##### `padding`  
*bool* | *str*  

Specifies whether to pad sequences during tokenization.  

---

##### `truncation`  
*bool*  

Specifies whether to truncate sequences during tokenization.  

---

#### Examples
```python
from liqfit.collators import NLICollator
from liqfit.datasets import NLIDataset
from torch.utils.data import DataLoader

dataset = NLIDataset(....)
collator = NLICollator(....)
dataloader = DataLoader(dataset, collate_fn=collator)

# OR

from transformers import Trainer
trainer = Trainer(train_dataset=dataset, data_collator=collator)
```

### Custom Collator <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/collators/base_collator.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.collators.NLICollator`
#### Parameters
##### `tokenizer`  
`AutoTokenizer`, *Callable*  

The tokenizer used to process input text into input IDs.  

---

##### `max_length`  
*int*  

Maximum length applied during tokenization of input sequences.  

---

##### `padding`  
*bool* | *str*  

Specifies whether to pad sequences during tokenization.  

---

##### `truncation`  
*bool*  

Specifies whether to truncate sequences during tokenization.  

---

#### Examples
The `Collator` base class here just groups your batch into one dictionary instead of a list of dictionaries.
```python
from liqfit.collators import Collator

class MyCollator(Collator):
    def __init__(self, tokenizer, max_length, padding, truncation)
        super().__init__(tokenizer, max_length, padding, truncation)
    
    def collate(self, batch):
        # your collate implementation.
```

## Datasets <sup><a href="https://github.com/Knowledgator/LiqFit/tree/main/src/liqfit/datasets" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---
### NLIDataset <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/datasets/nli_dataset.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.datasets.NLIDataset`
#### Parameters
##### `hypothesis`  
*List[str]*  

List of string sequences.  

---

##### `premises`  
*List[str]*  

List of string sequences.  

---

##### `labels`  
*List[int]*  

List of labels as integers.  

---
### load_dataset()
`liqfit.datasets.NLIDataset.load_dataset`
#### Parameters
##### `dataset`  
*Optional[Dataset]*  

Dataset object to use if `dataset_name` is not provided.  
(Defaults to `None`).  

---

##### `dataset_name`  
*int*  

Dataset name to load from Hugging Face datasets if `dataset` is not provided.  
(Defaults to `None`).  

---

##### `classes`  
*Optional[List[str]]*  

List of classes available in the dataset.  
(Defaults to `None`).  

---

##### `text_column`  
*Optional[str]*  

Name of the column containing the text data.  
(Defaults to `"text"`).  

---

##### `label_column`  
*Optional[str]*  

Name of the column containing the labels.  
(Defaults to `"label"`).  

---

##### `template`  
*Optional[str]*  

Template string used to concatenate the label.  
(Defaults to `"This example is {}."`).  

---

##### `normalize_negatives`  
*bool*  

Whether to normalize negative examples.  
(Defaults to `False`).  

---

##### `positives`  
*int*  

Label ID representing positive examples.  
(Defaults to `1`).  

---

##### `negatives`  
*int*  

Label ID representing negative examples.  
(Defaults to `-1`).  

---

##### `multi_label`  
*bool*  

Whether each example has more than one label.  
(Defaults to `False`).  

---

#### Examples
```python
from liqfit.datasets import NLIDataset
from datasets import load_dataset

nli_dataset = load_dataset("your/nli_dataset")

dataset = NLIDataset(
  hypothesis = nli_dataset['hypothesis'],
  premises = nli_dataset['premises'],
  labels = nli_dataset['labels'] # labels expected to be encoded.
)
                     
# OR

dataset = NLIDataset.load_dataset(nli_dataset, classes=["happiness", "sadness", ...])
```

## Losses <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/losses/losses.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---
### Focal Loss <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/losses/losses.py#L235" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`class liqfit.losses.FocalLoss`
#### Parameters
##### `ignore_index`  
*int*  

Index that will be ignored while calculating the loss.  

---

##### `alpha`  
*float*, optional  

Weighting factor between 0 and 1.  
(Defaults to `0.5`).  

---

##### `gamma`  
*float*, optional  

Focusing parameter where γ ≥ 0.  
(Defaults to `2.0`).  

---

##### `reduction`  
*str*  

Specifies the reduction method to apply to the output.  

---

#### Examples
```python
from liqfit.losses import FocalLoss
import torch
# FocalLoss in liqfit supports `ignore_index`
# parameter which could be used in token classification

x = torch.randn((1, 10, 20))
y = torch.randint(0, 10, (1, 10))
focal_loss = FocalLoss()
loss = focal_loss(x.view(-1, x.shape[-1]), y.view(-1))
```

### Binary Cross Entropy <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/losses/losses.py#L44" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`class liqfit.losses.BinaryCrossEntropyLoss`
#### Parameters
##### `multi_target`  
*bool*, optional  

Whether the labels are multi-target.  

---

##### `weight`  
*torch.Tensor*, optional  

Manual rescaling weight applied to the loss of each batch element.  

---

##### `reduction`  
*str*, optional  

Reduction method to apply on the loss.  
(Defaults to `"mean"`).  

---

#### Examples
Simple wrapper over PyTorch's binary_cross_entropy_with_logits loss function to support multi-target inputs
```python
from liqfit.losses import BinaryCrossEntropyLoss
import torch

x = torch.randn((1, 10, 20))
y = torch.randint(0, 2, (1, 10))
binary_loss = BinaryCrossEntropyLoss(multi_target=True)
loss = binary_loss(x, y) # No need for reshaping.
```

### Cross Entropy Loss <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/losses/losses.py#L127" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`class liqfit.losses.CrossEntropyLoss`
#### Parameters
##### `multi_target`  
*bool*, optional  

Whether the labels are multi-target.  

---

##### `weight`  
*torch.Tensor*, optional  

Manual rescaling weight applied to the loss of each batch element.  

---

##### `reduction`  
*str*, optional  

Reduction method to apply on the loss.  
(Defaults to `"mean"`).  

---

##### `ignore_index`  
*int*  

Index that will be ignored while calculating the loss.  

---

##### `label_smoothing`  
*float*, optional  

Value in [0.0, 1.0] specifying the amount of label smoothing.  
A value of `0.0` means no smoothing. Targets become a mix of the original label and a uniform distribution, as described in *Rethinking the Inception Architecture for Computer Vision*.  
(Defaults to `0.0`).  

---

#### Examples
Simple wrapper over PyTorch's cross_entropy loss function to support multi-target inputs
```python
from liqfit.losses import CrossEntropyLoss
import torch

x = torch.randn((1, 10, 20))
y = torch.randint(0, 2, (1, 10))
ce_loss = CrossEntropyLoss(multi_target=True)
loss = ce_loss(x, y) # No need for reshaping.
```

## Modeling <sup><a href="https://github.com/Knowledgator/LiqFit/tree/main/src/liqfit/modeling" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---
### LiqFitBackbone <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/modeling/backbone.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`classliqfit.modeling.LiqFitBackbone`
#### Parameters
##### `config`  
*PretrainedConfig*  

Backbone configuration.  

---

##### `backbone`  
*nn.Module*  

Pretrained model (backbone).  

---

##### `push_backbone_only`  
*bool*, optional  

Whether to push only the backbone model or the entire wrapped model to Hugging Face.  

---

#### Examples
If you want to customize your backbone model, you wrap your model inside LiqFitBackbone

```python
from liqfit.modeling import LiqFitBackbone
from transformers import AutoModel

class MyBackboneModel(LiqFitBackbone):
    def __init__(self):
        backbone_model = AutoModel.from_pretrained(...)
        super.__init__(backbone_model.config, backbone_model)
    
    def encode(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        return last_hidden_state
```
- If you want to add your own pooling layer.

```python
from liqfit.modeling.pooling import FirstTokenPooling1D

class MyBackboneModel(LiqFitBackbone):
    def __init__(self):
        backbone_model = AutoModel.from_pretrained(...)
        super.__init__(backbone_model.config, backbone_model)
        self.first_token_pooling = FirstTokenPooling1D()

    def encode(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        pooled_output = self.first_token_pooling(last_hidden_state)
        return pooled_output     
```
### LiqFitModel <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/modeling/model.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.modeling.Model`
#### Parameters
##### `config`  
*PretrainedConfig*  

Backbone configuration.  

---

##### `backbone`  
*nn.Module*  

Pretrained model (backbone).  

---

##### `head`  
*LiqFitHead*  

Downstream head.  

---

##### `loss_func`  
*Optional[nn.Module]*  

Loss function called after each forward pass if labels are provided.  
(Defaults to `None`).  

---

##### `normalize_backbone_embeddings`  
*bool*  

Whether to normalize the output embeddings from the backbone using `torch.nn.functional.normalize`.  
(Defaults to `False`).  

---

##### `labels_name`  
*str*  

Name of the labels parameter passed to the forward method.  

---

##### `push_backbone_only`  
*bool*, optional  

Whether to push only the backbone model or the entire wrapped model to Hugging Face.  

---

#### Examples
- Using `LiqFitModel` class with `transformers` library.

```python
from liqfit.modeling import LiqFitBackbone
from liqfit.modeling import LiqFitModel
from transformers import AutoModel

backbone_model = AutoModel.from_pretrained(...)
model = LiqFitModel(backbone_model.config, backbone_model)
```
- Using `LiqFitModel` with one of the available heads.

```python
from liqfit.modeling import LiqFitBackbone
from liqfit.modeling import LiqFitModel
from transformers import AutoModel

class MyBackboneModel(LiqFitBackbone):
    def __init__(self):
        backbone_model = AutoModel.from_pretrained(...)
        super.__init__(backbone_model.config, backbone_model)
    
    def encode(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        return last_hidden_state

backbone = MyBackboneModel()
head = ClassClassificationHead(backbone.config.hidden_size, 3, multi_target=True)

model = LiqFitModel(backbone.config, backbone, head)

x = torch.randint(0, 20, (1, 20))
out = model(x)
```
### LiqFitHead <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/heads.py#L11" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`class liqfit.modeling.LiqFitHead`

#### Examples
Creating custom Downstream head

```python
from liqfit.modeling.heads import LiqFitHead
from liqfit.modeling.heads import HeadOutput
from torch import nn

class MyOwnDownstreamHead(LiqFitHead):
    def __init__(in_features, out_features):
        self.linear = nn.Linear(in_features, out_features)

    def compute_loss(self, logits, labels):
        # your loss function implementation

    def forward(self, embeddings, labels=None):
        # your forward implementation.
        return HeadOutput(...)
```

### LabelClassificationHead <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/heads.py#L40" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.modeling.LabelClassificationHead`
#### Parameters
##### `in_features`  
*int*  

Number of input features.  

---

##### `out_features`  
*int*  

Number of output features.  

---

##### `multi_target`  
*bool*  

Whether the output is multi-target (used for loss calculation).  

---

##### `bias`  
*bool*  

Whether to use bias in the `nn.Linear` layer.  

---

##### `temperature`  
*int*  

Temperature used to calibrate the output by dividing the linear layer output.  
(Defaults to `1.0`).  

---

##### `eps`  
*float*  

Epsilon added to the temperature for numerical stability.  
(Defaults to `1e-5`).  

---

#### Examples
```python
from liqfit.modeling.heads import LabelClassificationHead
import torch

head = LabelClassificationHead(512, 20, multi_target=True)
embeddings = torch.randn((1, 10, 512))
output = head(embeddings)
```

### ClassClassificationHead <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/heads.py#L89" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.modeling.ClassClassificationHead`
#### Parameters
##### `in_features`  
*int*  

Number of input features.  

---

##### `out_features`  
*int*  

Number of output features.  

---

##### `multi_target`  
*bool*  

Whether the output is multi-target (used for loss calculation).  

---

##### `bias`  
*bool*  

Whether to use bias in the `nn.Linear` layer.  

---

##### `temperature`  
*int*  

Temperature used to calibrate the output by dividing the linear layer output.  
(Defaults to `1.0`).  

---

##### `eps`  
*float*  

Epsilon added to the temperature for numerical stability.  
(Defaults to `1e-5`).  

---

##### `ignore_index`  
*int*  

Index that will be ignored during loss calculation.  
(Defaults to `-100`).  

---

#### Examples
```python
from liqfit.modeling.heads import ClassClassificationHead
import torch

head = ClassClassificationHead(512, 20, multi_target=True)
embeddings = torch.randn((1, 10, 512))
output = head(embeddings)
```

### ClassificationHead <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/heads.py#L141" target="_blank" rel="noopener noreferrer">[source]</a></sup>

`class liqfit.modeling.ClassificationHead`
#### Parameters
##### `in_features`  
*int*  

Number of input features.  

---

##### `out_features`  
*int*  

Number of output features.  

---

##### `pooler`  
*nn.Module*  

Pooling function to use if the input is not multi-target.  

---

##### `loss_func`  
*nn.Module*  

Loss function called if labels are provided.  

---

##### `bias`  
*bool*  

Whether to use bias in the `nn.Linear` layer.  

---

##### `temperature`  
*int*  

Temperature used to calibrate the output by dividing the linear layer output.  
(Defaults to `1.0`).  

---

##### `eps`  
*float*  

Epsilon added to the temperature for numerical stability.  
(Defaults to `1e-5`).  

---

#### Examples
For more flexibility in passing your loss function and your pooling method.


```python
from liqfit.modeling.heads import ClassClassificationHead
import torch

head = ClassClassificationHead(512, 20, multi_target=True)
embeddings = torch.randn((1, 10, 512))
output = head(embeddings)
```

## Pooling Functions <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/modeling/pooling.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---

### GlobalMaxPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L7" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalMaxPooling1D`  
Applies global max pooling over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalMaxPooling1D
import torch

x = torch.randn((1, 10, 20))
pooler = GlobalMaxPooling1D()
out = pooler(x)
```

---

### GlobalAbsAvgPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L28" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalAbsAvgPooling1D`  
Applies global average pooling on the absolute values over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

##### `attention_mask`  
*torch.Tensor*  
Mask tensor of shape `(B, T)` to mask out padding tokens during pooling.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalAbsAvgPooling1D
import torch

x = torch.randn((1, 10, 20))
attention_mask = torch.ones((1, 10))
pooler = GlobalAbsAvgPooling1D()
out = pooler(x, attention_mask)
```

---

### GlobalAbsMaxPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L67" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalAbsMaxPooling1D`  
Applies global max pooling on the absolute values over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

##### `attention_mask`  
*torch.Tensor*  
Mask tensor of shape `(B, T)` to mask out padding tokens during pooling.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalAbsMaxPooling1D
import torch

x = torch.randn((1, 10, 20))
attention_mask = torch.ones((1, 10))
pooler = GlobalAbsMaxPooling1D()
out = pooler(x, attention_mask)
```

---

### GlobalRMSPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L53" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalRMSPooling1D`  
Applies global root mean square pooling over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

##### `attention_mask`  
*torch.Tensor*  
Mask tensor of shape `(B, T)` to mask out padding tokens during pooling.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalRMSPooling1D
import torch

x = torch.randn((1, 10, 20))
attention_mask = torch.ones((1, 10))
pooler = GlobalRMSPooling1D()
out = pooler(x, attention_mask)
```

---

### GlobalSumPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L44" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalSumPooling1D`  
Applies global sum pooling over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

##### `attention_mask`  
*torch.Tensor*  
Mask tensor of shape `(B, T)` to mask out padding tokens during pooling.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalSumPooling1D
import torch

x = torch.randn((1, 10, 20))
attention_mask = torch.ones((1, 10))
pooler = GlobalSumPooling1D()
out = pooler(x, attention_mask)
```

---

### GlobalAvgPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L28" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.GlobalAvgPooling1D`  
Applies global average pooling over the temporal dimension.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

##### `attention_mask`  
*torch.Tensor*  
Mask tensor of shape `(B, T)` to mask out padding tokens during pooling.

---

#### Example  
```python
from liqfit.modeling.pooling import GlobalAvgPooling1D
import torch

x = torch.randn((1, 10, 20))
attention_mask = torch.ones((1, 10))
pooler = GlobalAvgPooling1D()
out = pooler(x, attention_mask)
```

---

### FirstTokenPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L14" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.FirstTokenPooling1D`  
Selects the first token's embedding from the sequence.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

---

#### Example  
```python
from liqfit.modeling.pooling import FirstTokenPooling1D
import torch

x = torch.randn((1, 10, 20))
pooler = FirstTokenPooling1D()
out = pooler(x)
```

### LastTokenPooling1D <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/modeling/pooling.py#L21"  target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.modeling.pooling.LastTokenPooling1D`  
Selects the first token's embedding from the sequence.

#### Parameters  
##### `x`  
*torch.Tensor*  
Input tensor of shape `(B, T, E)`.

---

#### Example  
```python
from liqfit.modeling.pooling import LastTokenPooling1D
import torch

x = torch.randn((1, 10, 20))
pooler = LastTokenPooling1D()
out = pooler(x)
```

## Models <sup><a href="https://github.com/Knowledgator/LiqFit/tree/main/src/liqfit/models" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---

### DebertaConfigWithLoss <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/models/deberta.py#L34" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.models.deberta.DebertaConfigWithLoss`  
Extends the DeBERTa configuration to include loss-specific parameters.

#### Parameters  
##### `loss_type`  
*str*  
Specifies the loss function to be used when labels are provided.

##### `focal_loss_alpha`  
*float*, optional  
Weighting factor between 0 and 1.  
(Defaults to `0.5`).

##### `focal_loss_gamma`  
*float*, optional  
Focusing parameter where γ ≥ 0.  
(Defaults to `2.0`).

##### `**kwargs`  
Additional keyword arguments for the DeBERTa model configuration.

---

#### Example  
```python
from liqfit.models.deberta import DebertaConfigWithLoss

config = DebertaConfigWithLoss(
    loss_type="focal_loss",
    focal_loss_alpha=0.5,
    focal_loss_gamma=2.0,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12
)
```

---

### DebertaV2ForZeroShotClassification <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/models/deberta.py#L52" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.models.deberta.DebertaV2ForZeroShotClassification`  
DeBERTa model tailored for zero-shot classification tasks.

#### Parameters  
##### `config`  
*DebertaConfigWithLoss*  
Configuration object specifying model and loss parameters.

---

#### Example  
```python
from liqfit.models.deberta import DebertaConfigWithLoss, DebertaV2ForZeroShotClassification

config = DebertaConfigWithLoss(loss_type="focal_loss")
model = DebertaV2ForZeroShotClassification(config)
```

---

### T5ConfigWithLoss <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/models/t5.py#L33" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.models.t5.T5ConfigWithLoss`  
Extends the T5 configuration to include loss-specific parameters.

#### Parameters  
##### `loss_type`  
*str*  
Specifies the loss function to be used when labels are provided.

##### `focal_loss_alpha`  
*float*, optional  
Weighting factor between 0 and 1.  
(Defaults to `0.5`).

##### `focal_loss_gamma`  
*float*, optional  
Focusing parameter where γ ≥ 0.  
(Defaults to `2.0`).

##### `**kwargs`  
Additional keyword arguments for the T5 model configuration.

---

#### Example  
```python
from liqfit.models.t5 import T5ConfigWithLoss

config = T5ConfigWithLoss(
    loss_type="focal_loss",
    focal_loss_alpha=0.5,
    focal_loss_gamma=2.0,
    d_model=512,
    num_layers=6,
    num_heads=8
)
```

---

### T5ForZeroShotClassification <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/models/t5.py#L69" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.models.t5.T5ForZeroShotClassification`  
T5 model tailored for zero-shot classification tasks.

#### Parameters  
##### `config`  
*T5ConfigWithLoss*  
Configuration object specifying model and loss parameters.

---

#### Example  
```python
from liqfit.models.t5 import T5ConfigWithLoss, T5ForZeroShotClassification

config = T5ConfigWithLoss(loss_type="focal_loss")
model = T5ForZeroShotClassification(config)
```

---

## Pipelines <sup><a href="https://github.com/Knowledgator/LiqFit/blob/main/src/liqfit/pipeline/inference.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>

---

### ZeroShotClassificationPipeline <sup><a href="https://github.com/Knowledgator/LiqFit/blob/51ba2714813ae1cf110f7e600cd7f2663cdec39c/src/liqfit/pipeline/inference.py#L63" target="_blank" rel="noopener noreferrer">[source]</a></sup>  
`class liqfit.pipeline.ZeroShotClassificationPipeline`  
Facilitates zero-shot text classification using fine-tuned cross-encoder models.

#### Parameters  
##### `model`  
*AutoModelForSequenceClassification* | *CrossFitModel* | *torch.nn.Module*  
Specifies the fine-tuned model to be used in the pipeline.

##### `tokenizer`  
*AutoTokenizer*  
Tokenizer responsible for converting input text into tokens.

##### `hypothesis_template`  
*str*, optional  
Template for generating hypotheses.  
(Defaults to `'{}'`).

##### `hypothesis_first`  
*bool*, optional  
Determines whether to place the hypothesis before the premise.  
(Defaults to `False`).

##### `encoder_decoder`  
*bool*, optional  
Indicates if the model operates as an encoder-decoder architecture.  
(Defaults to `True`).

---

#### Example  
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from liqfit.pipeline import ZeroShotClassificationPipeline

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
template = 'This example is {}.'

model_path = 'knowledgator/comprehend_it-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = ZeroShotClassificationPipeline(
    model=model, 
    tokenizer=tokenizer, 
    hypothesis_template=template
)

results = classifier(sequence_to_classify, candidate_labels, multi_label=True)
print(results)
```

---

#### Binary Reranking Example  
```python
model_path = 'BAAI/bge-reranker-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = ZeroShotClassificationPipeline(
    model=model, 
    tokenizer=tokenizer, 
    hypothesis_template=template,
    hypothesis_first=False
)

results = classifier(sequence_to_classify, candidate_labels, multi_label=True)
print(results)
```

---

#### Encoder-Decoder Example  
```python
from liqfit.pipeline import ZeroShotClassificationPipeline
from liqfit.models import T5ForZeroShotClassification
from transformers import T5Tokenizer

model = T5ForZeroShotClassification.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
tokenizer = T5Tokenizer.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')

classifier = ZeroShotClassificationPipeline(
    model=model, 
    tokenizer=tokenizer,
    hypothesis_template='{}', 
    encoder_decoder=True
)
```

---
