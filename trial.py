from attribute_standardizer.attr_standardizer_class import AttrStandardizer

model = AttrStandardizer("BEDBASE")

results = model.standardize(pep ="geo/gse228815:default")

print(results)
