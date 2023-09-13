#!/bin/bash
# https://sylabs.io/guides/3.6/user-guide/quick_start.html
## Install singularity on CentOS machine


# Requirements
sudo yum update -y
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y epel-release
sudo yum install -y golang openssl-devel libuuid-devel libseccomp-devel squashfs-tools
sudo yum update -y

# Install singularity
VERSION=3.7
sudo yum install -y singularity-runtime singularity-$VERSION

'''
# add_to_sub_gid_uid.sh
sudo sh -c "echo `whoami`:100000:65536 >> /etc/subuid"
sudo sh -c "echo `whoami`:100000:65536 >> /etc/subgid"
'''


## EOF
sudo yum --showduplicates search singularity
