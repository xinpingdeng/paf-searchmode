#!/usr/bin/env python

import os

script_name = "fold_file_fil.py"
conf_fname  = "fold_file_fil.conf"

#script_name = "fold_file.py"
#conf_fname  = "fold_file.conf"

#script_name = "process_fil_dump.py"
#conf_fname  = "process_fil_dump.conf"
dir_name    = "/beegfs/DENG/docker/"
source_name = "J0332+5434"
dada_fname  = "2018-04-17-19:22:11.56868_0000000000000000.000000.dada"
gpu         = 1
visiblegpu  = 1
nvprof      = 1
memcheck    = 0

if nvprof == 1:
    com_line = "nvprof --profile-child-processes ./{:s} -a {:s} -b {:s} -c {:s} -d {:d} -e {:s} -f {:s} -g {:d}".format(script_name, conf_fname, dir_name, source_name, gpu, str(visiblegpu), dada_fname, memcheck)
else:
    com_line = "./{:s} -a {:s} -b {:s} -c {:s} -d {:d} -e {:s} -f {:s} -g {:d}".format(script_name, conf_fname, dir_name, source_name, gpu, str(visiblegpu), dada_fname, memcheck)

print com_line
os.system(com_line)

#os.system("nvprof --profile-child-processes ./fold_stream.py -a fold_stream.conf -b 0 -c 10 -d /beegfs/DENG/docker/ -e J0218+4232 -f all")

#os.system("nvprof --profile-child-processes ./fold_file.py -a fold_file.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")
#os.system("cuda-memcheck ./fold_file.py -a fold_file.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")

#os.system("nvprof --profile-child-processes ./fold_file_fil.py -a fold_file_fil.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")
#os.system("cuda-memcheck ./fold_file_fil.py -a fold_file_fil.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada")
