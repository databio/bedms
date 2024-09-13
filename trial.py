from bedms import AttrStandardizer

model = AttrStandardizer("ENCODE")

schemas = model.get_available_schemas()

print(schemas)

# results = model.standardize(pep="geo/gse178283:default")
results = model.standardize(pep="geo/gse228634:default")

print(results)
