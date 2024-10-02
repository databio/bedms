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
```python
from bedms import AttrStandardizer

model = AttrStandardizer("ENCODE")
results = model.standardize(pep="geo/gse228634:default")

assert results
```

### Training custom schemas
Training your custom schema is very easy with `BEDMS`. You would need two things to get started:
1. Training Sets
2. `training_config.yaml`

To instantiate `TrainStandardizer` class:

```python
from bedms.train import TrainStandardizer

trainer = TrainStandardizer("training_config.yaml")

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
For standardizing based on custom schema, you would require a `custom_config.yaml`.

```python
from bedms import AttrStandardizer

model = AttrStandardizer("CUSTOM", "custom_config.yaml")

results = model.standardize(pep="geo/gse228634:default")

assert results
```

### Available schemas
To see the available schemas, you can run:
```
from bedms.const import AVAILABLE_SCHEMAS
print(AVAILABLE_SCHEMAS)

# >> ['ENCODE', 'FAIRTRACKS', 'BEDBASE'] 

```

AVAILABLE_SCHEMAS is a list of available schemas that you can use to standardize your metadata.
