# bedmess

bedmess is a tool used to standardize genomics/epigenomics metadata based on a schema chosen by the user ( eg. ENCODE, FAIRTRACKS).


Presently, bedmess only provides standardization according to the ENCODE schema.


You can install the `attribute_standardizer` by:

```
pip install attribute-standardizer

```

## Usage

Using Python, this is how you can run `attribute_standardizer` :


```
from attribute_standardizer.attribute_standardizer import attr_standardizer

attr_standardizer(pep=/path/to/pep, schema="ENCODE")
```

You can use the format provided in the `trial.py` script in this repository as a reference. 