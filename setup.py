#!/usr/bin/env python3

from setuptools import setup

# Function to read the requirements from 'requirements.txt'
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(name='bitnet',
      version='0.0.0',
      description="Neural Nets can be as performant as needed even with 1.58 bits",
      author='Dario Cazzani, Aleks Yeganov',
      packages=['bitnet'],
      classifiers=[
        "TBD"
      ],
      install_requires=read_requirements(),
      python_requires='>=3.11',
      extras_require={
        'testing': [
            "pytest",
        ],
      },
      include_package_data=True)