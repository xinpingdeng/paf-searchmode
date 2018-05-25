#!/usr/bin/env python

import os

length  = 20
numa    = 0
memsize = 80000000000
psrname = 'J0218+4232'
cfname  = 'fold_stream.conf'
hdir    = '/home/pulsar/'
ddir    = '/beegfs/DENG/docker'
uid     = 50000
gid     = 50000
dname   = 'searchmode'

os.system('./launch_searchmode_pipeline.py -a {:f} -b {:d} -c {:d} -d {:s} -e {:s} -f {:s} -g {:s} -i {:d} -j {:d} -k {:s}'.format(length, numa, memsize, psrname, cfname, ddir, hdir, uid, gid, dname))
