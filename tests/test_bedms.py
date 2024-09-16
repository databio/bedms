from bedms import AttrStandardizer


class TestBEDMES:
    def test_bedmes(self):
        model = AttrStandardizer("ENCODE")
        results = model.standardize(pep="geo/gse228634:default")

        assert results
