from attribute_standardizer.attr_standardizer_class import AttrStandardizer

model = AttrStandardizer("ENCODE")

results = model.standardize(pep ="geo/gse178283:default")

print(results)
