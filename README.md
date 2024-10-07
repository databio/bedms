# BEDMS

BEDMS (BED Metadata Standardizer) is a tool desgined to standardize genomics and epigenomics metadata attributes according to user-selected schemas such as `ENCODE`, `FAIRTRACKS` and `BEDBASE`. BEDMS ensures consistency and FAIRness of metadata across different platforms. Additionally, users have the option to train their own standardizer model using a custom schema (`CUSTOM`), allowing for the standardization of attributes based on users' specific research requirements. 

## Installation

To install `bedms` use this command: 
```
pip install bedms
```
or install the latest version from the GitHub repository:
```
pip install git+https://github.com/databio/bedms.git
```

## Usage

### Standardizing based on available schemas

To choose the schema you want to standardize according to, please refer to the [HuggingFace repository](https://huggingface.co/databio/attribute-standardizer-model6). Based on the schema design `.yaml` files, you can select which schema best represents your attributes. In the example below, we have chosen `encode` schema. 

```python
from bedms import AttrStandardizer

model = AttrStandardizer(
    repo_id="databio/attribute-standardizer-model6", model_name="encode"
)
results = model.standardize(pep="geo/gse228634:default")

assert results
```

### Training custom schemas
Training your custom schema is very easy with `BEDMS`. You would need two things to get started:
1. Training Sets
2. `training_config.yaml`

To instantiate `TrainStandardizer` class:

```python
from bedms.train import AttrStandardizerTrainer

trainer = AttrStandardizerTrainer("training_config.yaml")

```
To load the datasets and encode them:

```python
trainer.load_encode_data()
```

To train the custom model:

```python
trainer.training()
```

To test the custom model:

```python
trainer.testing()
```

To generate visualizations such as Learning Curves, Confusion Matrices, and ROC Curve:

```python
trainer.plot_visualizations()
```

### Standardizing based on custom schema

For standardizing based on custom schema, your model should be on HuggingFace. The directory structure should follow the instructions mentioned on [HuggingFace](https://huggingface.co/databio/attribute-standardizer-model6). 

```python
from bedms import AttrStandardizer

model = AttrStandardizer(
    repo_id="name/of/your/hf/repo", model_name="model/name"
)
results = model.standardize(pep="geo/gse228634:default")

assert results
```