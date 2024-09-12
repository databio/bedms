from attribute_standardizer.attr_standardizer import AttrStandardizer

model = AttrStandardizer("ENCODE")

schemas = model.show_available_schemas()

print(schemas)

#results = model.standardize(pep="geo/gse178283:default")
results = model.standardize(pep="geo/gse228634:default") 

print(results)