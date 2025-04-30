#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="rbc-tracking-model",
    version="0.1.0",
    description="Red blood cell tracking toolkit for microvascular networks",
    author="Your Name",
    packages=["tracking_model"],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "igraph",
        "pyvista",
        "tqdm",
        "networkx",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
    ],
) 