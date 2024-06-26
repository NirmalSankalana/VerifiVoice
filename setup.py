from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="verifyvoice",
    version="0.1.33",
    description="A package for verifying the voice of a person",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NirmalSankalana/VerifiVoice",
    author="Nirmal Sankalana, Nipun Thejan",
    author_email="sankalana.nirmal@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["verifyvoice"],
    include_package_data=True,
    install_requires=["numpy >=1.2", "soundfile >=0.12.1", "webrtcvad==2.0.10", "librosa"],
)
