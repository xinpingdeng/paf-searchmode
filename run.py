#!/usr/bin/env python

import os

#os.system("nvprof --profile-child-processes ./fold_stream.py -a fold_stream.conf -b 0 -c 10 -d /beegfs/DENG/docker/ -e J0218+4232 -f all")
#os.system("nvprof --profile-child-processes ./fold_file.py -a fold_file.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")
os.system("nvprof --profile-child-processes ./fold_file_fil.py -a fold_file.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")
