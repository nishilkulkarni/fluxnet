from setuptools import setup, find_packages

setup(
    name="fluxnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torch-geometric>=2.0.0",
    ],
    author="Nishil Kulkarni",
    author_email="nishilkulk@gmail.com",
    description="A graph neural network library featuring continuous kernel convolutions and self-attention mechanism",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nishilkulkarni/fluxnet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)