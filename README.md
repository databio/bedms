# bedmess

bedmess is a tool used to standardize genomics/epigenomics metadata based on a schema chosen by the user ( eg. ENCODE, FAIRTRACKS).


Presently, bedmess only provides standardization according to the ENCODE schema.


To standardize a PEP, add the PEPhub registry path to the `trial.py` file. 


Assuming you are in the bedmess directory, you can modift `trial.py` as:

```
from attribute_standardizer.attribute_standardizer import attr_standardizer

attr_standardizer(pep=/path/to/pep, schema="ENCODE")
```
