# Chemical Models
## 🧪 Chemical-Converters

[**Chemical-Converters**](https://github.com/Knowledgator/chemical-converters) is a Python library developed by Knowledgator for translating between different chemical formats, specifically between SMILES (Simplified Molecular Input Line Entry System) strings and IUPAC (International Union of Pure and Applied Chemistry) names. This tool is designed to assist researchers, students, and professionals in the chemical domain by providing accurate and efficient conversions.

---

### Features

- **Bidirectional Conversion**: Translate between SMILES and IUPAC names.
- **Multiple Models**: Choose from various models optimized for different accuracy and performance needs.
- **Style Preferences**: Specify preferred IUPAC naming styles using style tokens.
- **Batch Processing**: Efficiently process large datasets with batch conversion.
- **Validation**: Validate conversions using Tanimoto similarity scores.

---

### Installation

Install the library using pip:

```bash
pip install chemical-converters
```

---

### Models

The models are based on Google's MT5 architecture, modified to support chemical format translations. Available models include:

| Model Name                              | Accuracy | Size (MB) | Task             |
|-----------------------------------------|----------|-----------|------------------|
| `SMILES2IUPAC-canonical-small`          | 75%      | 24        | SMILES to IUPAC  |
| `SMILES2IUPAC-canonical-base`           | 86.9%    | 180       | SMILES to IUPAC  |
| `IUPAC2SMILES-canonical-small`          | 88.9%    | 24        | IUPAC to SMILES  |
| `IUPAC2SMILES-canonical-base`           | 93.7%    | 180       | IUPAC to SMILES  |

To check available models:

```python
from chemicalconverters import NamesConverter

print(NamesConverter.available_models())
```

---

### Usage

#### SMILES to IUPAC

```python
from chemicalconverters import NamesConverter

converter = NamesConverter(model_name="knowledgator/SMILES2IUPAC-canonical-base")
result = converter.smiles_to_iupac('CCO')
print(result)  # Output: ['ethanol']
```

##### Specifying IUPAC Style

Use style tokens to specify the desired IUPAC naming convention:

- `<BASE>`: Common name, possibly a mix of traditional and systematic styles.
- `<SYST>`: Fully systematic name without trivial names.
- `<TRAD>`: Name based on trivial names of substance parts.

Example:

```python
converter.smiles_to_iupac(['<SYST>CCO', '<TRAD>CCO', '<BASE>CCO'])
# Output: ['ethanol', 'ethanol', 'ethanol']
```

##### Batch Processing

```python
converter.smiles_to_iupac(
    ["<BASE>C=CC=C" for _ in range(10)],
    num_beams=1,
    process_in_batch=True,
    batch_size=1000
)
# Output: ['buta-1,3-diene', 'buta-1,3-diene', ...]
```

##### Validation

Validate the conversion by checking the Tanimoto similarity between the original and converted molecules:

```python
converter.smiles_to_iupac('CCO', validate=True)
# Output: ['ethanol'], 1.0
```

Manual validation:

```python
validation_model = NamesConverter(model_name="knowledgator/IUPAC2SMILES-canonical-base")
NamesConverter.validate_iupac(input_sequence='CCO', predicted_sequence='ethanol', validation_model=validation_model)
# Output: 1.0
```

*Note: Validation is not implemented for batch processing.*

#### IUPAC to SMILES

```python
converter = NamesConverter(model_name="knowledgator/IUPAC2SMILES-canonical-base")
result = converter.iupac_to_smiles('ethanol')
print(result)  # Output: ['CCO']
```

##### Batch Processing

```python
converter.iupac_to_smiles(
    ["buta-1,3-diene" for _ in range(10)],
    num_beams=1,
    process_in_batch=True,
    batch_size=1000
)
# Output: ['<SYST>C=CC=C', '<SYST>C=CC=C', ...]
```

---

*Remember, chemistry is not just about reactions; it's about connections. Let's build those connections together!*
