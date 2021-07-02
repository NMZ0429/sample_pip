import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="namazu",
    version="0.0.1",
    author="NMZ",
    author_email="gen0429@icloud.com",
    description="Utility library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NMZ0429/sample_pip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch", "torchmetrics"],
)
