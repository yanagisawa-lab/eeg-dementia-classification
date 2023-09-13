#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2021-09-15 07:13:45 (ywatanabe)"

from setuptools import setup
from codecs import open
from os import path
import re

################################################################################
PACKAGE_NAME = "MYPACKAGE"
DESCRIPTION = "THIS is my package."
KEYWORDS = ["keyword_1", "keyword_2"]
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]
################################################################################

root_dir = path.abspath(path.dirname(__file__))


def _requirements():
    return [
        name.rstrip()
        for name in open(path.join(root_dir, "requirements.txt")).readlines()
    ]


def _test_requirements():
    return [
        name.rstrip()
        for name in open(path.join(root_dir, "test-requirements.txt")).readlines()
    ]


with open(path.join(root_dir, PACKAGE_NAME, "__init__.py")) as f:
    init_text = f.read()
    version = re.search(r"__version__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)
    license = re.search(r"__license__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)
    author = re.search(r"__author__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)
    author_email = re.search(
        r"__author_email__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    url = re.search(r"__url__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)

assert version
assert license
assert author
assert author_email
assert url

with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=PACKAGE_NAME,
    packages=[PACKAGE_NAME],
    version=version,
    license=license,
    install_requires=_requirements(),
    tests_require=_test_requirements(),
    author=author,
    author_email=author_email,
    url=url,
    description=DESCRIPTION,
    long_description=long_description,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    python_requires=">=3.0",
)
