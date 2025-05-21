from setuptools import setup, find_packages

setup(
    name="kv-compression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pyyaml",
        "psutil",
        "tensorly",
        "pytest",
        "transformers",
        "datasets",
    ],
)
