#!/bin/bash

# https://sylabs.io/guides/3.6/user-guide/quick_start.html
## Install singularity 3.6.3 on CentOS 8 host machine

sudo yum remove singularity -y # when you previously installed singularity using yum.
# ====================================================================================================
#  Package                   Architecture         Version                   Repository           Size
# ====================================================================================================
# Removing:
#  singularity               x86_64               3.7.1-1.el8               @epel               138 M

# Transaction Summary
# ====================================================================================================
# Remove  1 Package


## Prepare requirements
sudo yum update -y
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y epel-release
sudo yum install -y golang openssl-devel libuuid-devel libseccomp-devel squashfs-tools
sudo yum update -y


## Download the singularity source
cd /tmp
VERSION=3.6.3
wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz
tar xvf singularity-${VERSION}.tar.gz
rm -f singularity-${VERSION}.tar.gz


## Build and install a designated version of singularity
PREFIX=/opt
cd /tmp/singularity
N_CPUS=10
sudo ./mconfig --prefix=$PREFIX
cd builddir
sudo make -j $N_CPUS
sudo make install -j $N_CPUS
PATH=$PREFIX/bin:$PATH
echo Singularity `singularity version` has been installed in `which singularity`.

echo Please add \"$PREFIX/bin\" to the shell variable \"PATH\"
echo The following two lines would be useful to use \"the --fakeroot option\".
echo sudo sh -c "echo `whoami`:100000:65536 >> /etc/subuid"
echo sudo sh -c "echo `whoami`:100000:65536 >> /etc/subgid"

## EOF
