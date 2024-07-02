from setuptools import setup, find_packages

setup(
    name = 'bedmess',
    version = '1.0',
    packages = find_packages(),
    install_requires = [
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'pephubclient'
    ],
    author = 'Databio',
    author_email= None,
    description = 'BEDMess for metadata attribute standardization',
    long_description = None,
    license = None,
)