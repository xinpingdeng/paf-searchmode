#!/usr/bin/env python

# docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --rm -it --ulimit memlock=40000000000 -v host_dir:doc_dir --net=host xinpingdeng/fold_mode
# docker run --runtime=nvidia tells the container that we will use nvidia/cuda library at runtime;
# --rm means the container will be released once it finishs;
# -i means it is a interactive container;
# -t oallocate a pseudo-TTY;
# as single character can be combined, we use -it instead of -i and -t here;
# --ulimit memlock=XX tell container to use XX bytes locked shared memory, NOTE, here is the size of shared memory used by all docker containers on host machine, not the current one;
# --net=host let container to use host network configuration;
# -e NVIDIA_DRIVER_CAPABILITIES controls which driver libraries/binaries will be mounted inside the container;
# -e NVIDIA_VISIBLE_DEVICES which GPUs will be made accessible inside the container;
# -v maps the host directory with the directory inside container, if the directories do not exist, docker will create them;
# Detail on how to setup nvidia docker image can be found at https://github.com/NVIDIA/nvidia-container-runtime;

import os, argparse

# ./launch_searchmode_pipeline.py -a 20 -b 0 -c 80000000000 -d J0218+4232 -e fold_stream.conf -f /beegfs/DENG/docker -g /home/pulsar -i /home/pulsar/paf-searchmode -j 50000 -k 50000 -l searchmode
# Read in command line arguments
parser = argparse.ArgumentParser(description='Launch the pipeline to catpure and fold data stream from BMF')
parser.add_argument('-a', '--length', type=float, nargs='+',
                    help='The length in second for data capture')
parser.add_argument('-b', '--numa', type=int, nargs='+',
                    help='On which numa node we do the work, 0 or 1')
parser.add_argument('-c', '--memsize', type=int, nargs='+',
                    help='Bytes of shared memory used by all docker container running on the host')
parser.add_argument('-d', '--psrname', type=str, nargs='+',
                    help='The name of pulsar')
parser.add_argument('-e', '--cfname', type=str, nargs='+',
                    help='The name of configuration file')
parser.add_argument('-f', '--ddir', type=str, nargs='+',
                    help='Directory with configuration file, timing model and to record data')
parser.add_argument('-g', '--sdir', type=str, nargs='+',
                    help='Source directory')
parser.add_argument('-i', '--uid', type=int, nargs='+',
                    help='UID of user')
parser.add_argument('-j', '--gid', type=int, nargs='+',
                    help='Group ID of the user belongs to')
parser.add_argument('-k', '--dname', type=str, nargs='+',
                    help='The name of docker container')

args    = parser.parse_args()
length  = args.length[0]
numa    = args.numa[0]    # It determines numa node id, memory allocation place, NiC id and also GPU id
memsize = args.memsize[0] # The number here should be larger than the required shared memory size in bytes, the required shared memory size is the total shared memory used by all containers. 
psrname = args.psrname[0]
cfname  = args.cfname[0]
ddir    = args.ddir[0]
sdir    = args.sdir[0]
uid     = args.uid[0]
gid     = args.gid[0]
dname   = args.dname[0]

gpu     = numa  # Either numa or "all"
dvolume = '{:s}:{:s}'.format(ddir, ddir)
svolume = '{:s}:{:s}'.format(sdir, sdir)

com_line = "docker run -it --runtime=nvidia -v {:s} -v {:s} -v /home/pulsar/.Xauthority:/home/pulsar/.Xauthority -e DISPLAY -u {:d}:{:d} -e NVIDIA_VISIBLE_DEVICES={:s} -e NVIDIA_DRIVER_CAPABILITIES=all --rm --ulimit memlock={:d} --net=host --name {:s} searchmode".format(dvolume, svolume, uid, gid, str(gpu), memsize, dname)

print com_line
print "\nYou are going to a docker container with the name {:s}!\n".format(dname)

os.system(com_line)
