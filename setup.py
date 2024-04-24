from setuptools import setup, find_packages

setup(
    name='virifyvoice',
    version='0.1.0',
    description='python package for text independent speaker verification',
    author='Nirmal Sankalana, Nipun Fonseka',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=['numpy', 'soundfile', 'onnxruntime']
)