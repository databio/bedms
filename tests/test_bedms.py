from bedms import AttrStandardizer


class TestBEDMES:
    def test_bedmes(self):
        model = AttrStandardizer(repo_id='databio/attribute-standardizer-model6', model_name='encode')
        results = model.standardize(pep="geo/gse228634:default")

        assert results
