#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# name:      setup.py
# author:    Pengjia Zhu <zhupengjia@gmail.com>

import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "elsabot",
    version = "0.2.0",
    python_requires=">=3.0",
    author = "Pengjia Zhu",
    author_email = "zhupengjia@gmail.com",
    description = "elsabot",
    license = "MIT License",
    keywords = "Elsa Chatbot",
    url = "",
    packages= find_packages(),
    install_requires=required,
    scripts=[
        "elsa_interact.py",
        "elsa_train.py",
        "elsa_train_rl.py"
    ],
    long_description=read('README.md'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
)
