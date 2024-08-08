# bedmess

bedmess is a tool used to standardize genomics/epigenomics metadata based on a schema chosen by the user ( eg. ENCODE, FAIRTRACKS).


To install `attribute-standardizer` , you need to clone this repository first. Follow the steps given below to install:

```
git clone https://github.com/databio/bedmess.git

cd bedmess

pip install .

```

## Usage

Using Python, this is how you can run `attribute_standardizer` :


```
from attribute_standardizer.attribute_standardizer import attr_standardizer

attr_standardizer(pep=/path/to/pep, schema="ENCODE")
```

You can use the format provided in the `trial.py` script in this repository as a reference. 