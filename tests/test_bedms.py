import pytest
from bedms import AttrStandardizer


class TestBEDMES:
    def test_bedmes(self):

        model = AttrStandardizer("ENCODE")

        schemas = model.get_available_schemas()

        assert schemas
        # results = model.standardize(pep="geo/gse178283:default")
        results = model.standardize(pep="geo/gse228634:default")

        assert results
