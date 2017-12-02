#!/bin/sh
# This script is untested.
#
# On Windows, you'll have to download these files on your own,
# for example with your browser,
# and use for example 7zip to uncompress them.
#
# In all cases you should have in the end in the current folder:
# - train-images-idx3-ubyte
# - train-labels-idx1-ubyte
# - t10k-images-idx3-ubyte
# - t10k-labels-idx1-ubyte

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
