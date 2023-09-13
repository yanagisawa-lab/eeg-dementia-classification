#!/bin/bash

rm -rf build dist/* src/eeg_dementia_classification_MNet.egg-info
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*

## EOF
