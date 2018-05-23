#!/usr/bin/env python

import os

length  = 20
numa    = 0
memsize = 80000000000
psrname = 'J0218+4232'
cfname  = 'fold_stream.conf'
ddir    = '/beegfs/DENG/docker'
sdir    = '/home/pulsar/paf-searchmode'
uid     = 50000
gid     = 50000
dname   = 'searchmode'

os.system('./launch_searchmode_pipeline.py -l {:f} -n {:d} -m {:d} -p {:s} -c {:s} -d {:s} -s {:s} -u {:d} -g {:d} -a {:s}'.format(length, numa, memsize, psrname, cfname, ddir, sdir, uid, gid, dname))
