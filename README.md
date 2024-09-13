# BEDMS

BEDMS (BED Metadata Standardizer) is a tool used to standardize genomics/epigenomics metadata based on a schema chosen by the user ( eg. ENCODE, FAIRTRACKS, BEDBASE).


To install `attribute-standardizer` , you need to clone this repository first. Follow the steps given below to install:

```
git clone https://github.com/databio/bedms.git

cd bedms

pip install .

```

## Usage

Using Python, this is how you can run `attribute_standardizer` and print the results :


```
from attribute_standardizer import AttrStandardizer

model = AttrStandardizer("ENCODE")
model = AttrStandardizer("FAIRTRACKS")

results = model.standardize(pep ="geo/gse178283:default")

print(results)

```

To see the available schemas, you can run:
```
schemas = model.show_available_schemas()

print(schemas)
```

This will print the available schemas as a list. 

You can use the format provided in the `trial.py` script in this repository as a reference. 