# BEDMS

BEDMS (BED Metadata Standardizer) is a tool used to standardize genomics/epigenomics metadata based on a schema chosen by the user ( eg. ENCODE, FAIRTRACKS, BEDBASE).


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

```python
from bedms import AttrStandardizer

model = AttrStandardizer("ENCODE")
results = model.standardize(pep="geo/gse228634:default")

assert results
```


To see the available schemas, you can run:
```
from bedms.constants import AVAILABLE_SCHEMAS
print(AVAILABLE_SCHEMAS)

# >> ['ENCODE', 'FAIRTRACKS', 'BEDBASE'] 

```
AVAILABLE_SCHEMAS is a list of available schemas that you can use to standardize your metadata.
