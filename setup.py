#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_namespace_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

# with open('docs/HISTORY.md') as history_file:
#     history = history_file.read()

requirements = [
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pandas",
    "pynapple",
    "nemos",
    "dandi",
    "pynwb",
    "pooch",
    "mat73",
    "ipython",
    "jupyter",
]

setup(
    author="Sarah Jo Venditto",
    author_email="svenditto@flatironinstitute.org",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
    ],
    description="Neuroscience data analysis tutorials by NeuroStatsLab using publicly available, curated data sets",
    install_requires=requirements,
    license="MIT License",
    # long_description='pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.'
    # + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords=["neuroscience", "statistics"],
    name="nsl-tutorials",
    packages=find_namespace_packages(
        include=["nsl-tutorials"],
    ),
    url="https://github.com/neurostatslab/nsl-tutorials",
    version="v0.0.1",
    zip_safe=False,
    long_description_content_type="text/markdown",
    download_url="https://github.com/neurostatslab/nsl-tutorials/archive/refs/heads/main.zip",
)
