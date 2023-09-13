#!/bin/bash

################################################################################
## Installation
## 1. Put this file at $HOME as ~/singularity-aliases.bash
## 2. Add the line ". $HOME/singularity-aliases.bash" # automatically load this file
## 3. Put your sif file or sandbox dir under singularity dir of your project
## ./singularity
## ├── YOUR_REAL_SIF_FILE.sif
## ├── YOUR_REAL_SANDBOX_DIR
## ├── image.sif -> YOUR_REAL_SIF_FILE.sif (symlink)
## └── image -> YOUR_REAL_SANDBOX_DIR (symlink)
################################################################################

Usage examples:
$ sshell     # = singularity shell ./singularity/image.sif


################################################################################
## Working with .sif file (-> image.sif)
################################################################################
# shell
function sshell() {
    echo singularity shell --nv --cleanenv ./singularity/image.sif
    singularity shell --nv --cleanenv ./singularity/image.sif
}

# python
function spy () {
    echo "singularity exec --nv --cleanenv ./singularity/image.sif python3 $@" &&
    singularity exec --nv --cleanenv ./singularity/image.sif python3 $@
}

# ipython
function sipy () {
    echo "singularity exec --nv --cleanenv ./singularity/image.sif ipython $@" &&
    singularity exec --nv --cleanenv ./singularity/image.sif ipython $@
}

# jupyter
function sjpy () {
    echo "singularity exec --nv --cleanenv ./singularity/image.sif jupyter-notebook $@" &&
    singularity exec --nv --cleanenv ./singularity/image.sif jupyter-notebook $@
}

################################################################################
## Working with sandbox (= writable) directories (-> image)
################################################################################
# shell
function sshellw() {
    
    echo singularity shell --nv --cleanenv ./singularity/image
    singularity shell --nv --cleanenv ./singularity/image
}

# python
function spyw () {
    echo "singularity exec --nv --cleanenv ./singularity/image python3 $@" &&
    singularity exec --nv --cleanenv ./singularity/image python3 $@
}

# ipython
function sipyw () {
    echo "singularity exec --nv --cleanenv ./singularity/image ipython $@" &&
    singularity exec --nv --cleanenv ./singularity/image ipython $@
}

# jupyter-notebook
function sjpyw () {
    echo "singularity exec --nv --cleanenv ./singularity/image jupyter-notebook $@" &&
    singularity exec --nv --cleanenv ./singularity/image jupyter-notebook $@
}

################################################################################
## Build commands
################################################################################
# sif
function sbuild () {
    DEF=$1
    OPTION=$2

    FNAME=`echo $DEF | cut -d . -f 1`
    SIF=${FNAME}.sif

    echo singularity build $OPTION $SIF $DEF
    singularity build $OPTION $SIF $DEF
    
}

# sandbox with the singularity remote server
function sbuildr () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --remote $FNAME $DEF
    singularity build --remote $FNAME $DEF
}

# sandbox (= writable)
function sbuildw () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --fakeroot --sandbox $FNAME $DEF
    singularity build --fakeroot --sandbox $FNAME $DEF
}

# sandbox with the singularity remote server
function sbuildwr () {
    DEF=$1
    FNAME="${DEF%.*}"

    echo singularity build --remote --sandbox $FNAME $DEF
    singularity build --remote --sandbox $FNAME $DEF
}

## EOF
