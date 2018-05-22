#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

repeat     = 1
nsamp      = 1024
nchan      = 256
npol       = 2
ndim       = 2
ndata      = nsamp * nchan * npol * ndim
hdrsize    = 4096
dsize = 1
fdir  = "/beegfs/DENG/"
fname = '2018-03-29-12:37:55.212043_0000000000000000.000000.dada'
#fname = '2018-03-29-12:37:55.212043_0000032768000000.000000.dada'

blksize  = ndata * dsize
fname = os.path.join(fdir, fname)
f = open(fname, "r")
f.seek(hdrsize + repeat * blksize)
sample = np.array(np.fromstring(f.read(blksize), dtype='int8'))
#sample = np.reshape(sample, (nsamp, nchan, npol * ndim))

plt.figure()
plt.plot(sample)
plt.show()
