#!/usr/bin/python3

from setuptools import setup, find_packages

setup(
    name='p1afempy',
    version='0.1.0',
    description='Adaptive P1 FEM algorithms',
    author='Raphael Leu',
    author_email='raphaelleu95@gmail.com',
    packages=find_packages(exclude=("tests",)),
    install_requires=[
      "numpy",
      "scipy",
      "pathlib",
      "matplotlib"
    ]
)
