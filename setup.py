import os
import sys

from setuptools import find_packages, setup

PACKAGE_NAME = "attribute_standardizer"

# Ordinary dependencies
DEPENDENCIES = []
with open("requirements/requirements-all.txt", "r") as reqs_file:
    for line in reqs_file:
        if not line.strip():
            continue
        # DEPENDENCIES.append(line.split("=")[0].rstrip("<>"))
        DEPENDENCIES.append(line)

# Additional keyword arguments for setup().
extra = {"install_requires": DEPENDENCIES}


# Additional files to include with package
def get_static(name, condition=None):
    static = [
        os.path.join(name, f)
        for f in os.listdir(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
        )
    ]
    if condition is None:
        return static
    else:
        return [i for i in filter(lambda x: eval(condition), static)]


with open(f"{PACKAGE_NAME}/_version.py", "r") as versionfile:
    version = versionfile.readline().split()[-1].strip("\"'\n")

with open("README.md") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    version=version,
    description="BEDMess attribute standardizer for metadata attribute standardization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="project, metadata, bioinformatics",
    url="https://github.com/databio/bedmess/",
    author="Saanika Tambe",
    license="BSD2",
    include_package_data=True,
    # tests_require=(["pytest"]),
    setup_requires=(
        ["pytest-runner"] if {"test", "pytest", "ptr"} & set(sys.argv) else []
    ),
    **extra,
)
