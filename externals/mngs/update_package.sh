#!/bin/bash

# dir_orig=`pwd`
# mngs_root='/mnt/md0/mngs/'
# cd $mngs_root
rm -rf build dist/* src/mngs.egg-info
# pip uninstall mngs -y
# pip install -e .
# python3 -m pytest test
python3 setup.py sdist bdist_wheel
# twine upload -r testpypi dist/*
twine upload -r pypi dist/*

# pip install --no-cache-dir --upgrade ./dist/mngs-*-py3-none-any.whl --force-reinstall

# pip install mngs --upgrade
# cd $dir_orig
## EOF
