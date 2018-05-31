#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

repeat     = 1
nsamp      = 1
nchan      = 1024
npol       = 1
ndim       = 1
ndata      = nsamp * nchan * npol * ndim
hdrsize    = 4096
dsize = 1
fdir  = "/beegfs/DENG/docker"
fname = '2018-04-17-19:22:11_0000000000000000.000000.dada'

blksize  = ndata * dsize
fname = os.path.join(fdir, fname)
f = open(fname, "r")
f.seek(hdrsize + repeat * blksize)
sample = np.array(np.fromstring(f.read(blksize), dtype='uint8'))
#sample = np.reshape(sample, (nsamp, nchan, npol * ndim))

plt.figure()
plt.plot(sample)
plt.show()

f.close()
