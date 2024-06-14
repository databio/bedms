# bedmess

bedmess is a tool that can be used to standardize metadata according to the bedbase metadata output schema.

## Implementation
You can provide your metadata as a tsv file. Additionally, you have to choose between FAIRTRACS and ENCODE schemas.
Usage: Assuming you are in the scripts directory of bedmess:
   ```bash
      python3 attr_standardizer_cli.py --path PEPhub_registry / LOCAL --schema ENCODE / FAIRTRACKS /path/to/csv
   ```

Presently, we are only providing standardization according to the FAIRTRACKS schema. 
