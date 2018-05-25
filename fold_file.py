#!/usr/bin/env python

# ./fold_file.py -a fold_file.conf -b /beegfs/DENG/docker/ -c J0332+5434 -d 0 -e 0 -f 2018-04-17-19:22:11.56868_0000000000000000.000000.dada
# tempo2 -f mypar.par -pred "sitename mjd1 mjd2 freq1 freq2 ntimecoeff nfreqcoeff seg_length"

import os, time, threading, ConfigParser, argparse, socket, json, struct, sys

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

## Get center frequency from multi cast
#multicast_group = '224.1.1.1'
#server_address = ('', 5007)
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create the socket
#sock.bind(server_address) # Bind to the server address
#group = socket.inet_aton(multicast_group)
#mreq = struct.pack('4sL', group, socket.INADDR_ANY)
#sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)  # Tell the operating system to add the socket to the multicast group on all interfaces.
##sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, mreq)  
#pkt, address = sock.recvfrom(1<<16)
#data = json.loads(pkt)#['beams_direction']#['beam01']
##sock.shutdown(socket.SHUT_RDWR)
#sock.close()
#freq = float(data['sky_frequency'])
#print "The centre frequency is {:.1f}MHz".format(freq)

freq   = 1340.5  # it should be the value from main startup GUI of TOS plus 0.5
stream = 0       # Now we are folding files

# Read in command line arguments
parser = argparse.ArgumentParser(description='Fold data from DADA file')
parser.add_argument('-a', '--cfname', type=str, nargs='+',
                    help='The name of configuration file')
parser.add_argument('-b', '--directory', type=str, nargs='+',
                    help='In which directory we record the data and read configuration files and parameter files')
parser.add_argument('-c', '--psrname', type=str, nargs='+',
                    help='The name of pulsar')
parser.add_argument('-d', '--gpu', type=int, nargs='+',
                    help='The index of GPU')
parser.add_argument('-e', '--visiblegpu', type=str, nargs='+',
                    help='Visible GPU, the parameter is for the usage inside docker container.')
parser.add_argument('-f', '--dfname', type=str, nargs='+',
                    help='The name of data file.')

args         = parser.parse_args()
cfname       = args.cfname[0]
gpu          = args.gpu[0]
directory    = args.directory[0]
psrname      = args.psrname[0]
dfname       = args.dfname[0]

if(args.visiblegpu[0]==''):
    multi_gpu = 1;
if(args.visiblegpu[0]=='all'):
    multi_gpu = 1;
else:
    multi_gpu = 0;
    
# Play with configuration file
Config = ConfigParser.ConfigParser()
Config.read(cfname)

# Basic configuration
nsamp_df     = int(ConfigSectionMap("BasicConf")['nsamp_df'])
npol_samp    = int(ConfigSectionMap("BasicConf")['npol_samp'])
ndim_pol     = int(ConfigSectionMap("BasicConf")['ndim_pol'])
nchk_nic     = int(ConfigSectionMap("BasicConf")['nchk_nic'])

# Diskdb configuration
diskdb_ndf 	= int(ConfigSectionMap("DiskdbConf")['ndf'])
diskdb_nbuf    = ConfigSectionMap("DiskdbConf")['nblk']
diskdb_key     = ConfigSectionMap("DiskdbConf")['key']
diskdb_key     = format(int("0x{:s}".format(diskdb_key), 0), 'x')
diskdb_kfname  = "{:s}.key".format(ConfigSectionMap("DiskdbConf")['kfname_prefix'])
diskdb_hfname  = ConfigSectionMap("DiskdbConf")['hfname']
diskdb_nreader = ConfigSectionMap("DiskdbConf")['nreader']
diskdb_sod     = ConfigSectionMap("DiskdbConf")['sod']
diskdb_rbufsz  = diskdb_ndf *  nchk_nic * 7168
diskdb_cpu     = 0

# Process configuration
process_key        = ConfigSectionMap("ProcessConf")['key']
process_kfname     = "{:s}.key".format(ConfigSectionMap("ProcessConf")['kfname_prefix'])
process_key        = format(int("0x{:s}".format(process_key), 0), 'x')
process_sod        = ConfigSectionMap("ProcessConf")['sod']
process_nreader    = ConfigSectionMap("ProcessConf")['nreader']
process_nbuf       = ConfigSectionMap("ProcessConf")['nblk']
process_ndf        = int(ConfigSectionMap("ProcessConf")['stream_ndfstp'])
process_nstream    = int(ConfigSectionMap("ProcessConf")['nstream'])
process_osampratei = float(ConfigSectionMap("ProcessConf")['osamp_ratei'])
process_nchanratei = float(ConfigSectionMap("ProcessConf")['nchan_ratei'])
process_rbufsz     = int(0.5 * diskdb_rbufsz * process_osampratei / process_nchanratei)

process_hfname     = ConfigSectionMap("ProcessConf")['hfname']
process_cpu        = 1 

# Fold configuration
fold_cpu = 2
subint   = int(ConfigSectionMap("FoldConf")['subint'])

# Check the buffer block can be covered with multiple run of multiple streams
if (diskdb_ndf % (process_nstream * process_ndf)):
    print "Multiple run of multiple streams can not cover single ring buffer block, please edit configuration file {:s}".format(cfname)
    exit()
else:
    nrun_blk = diskdb_ndf / (process_nstream * process_ndf)

def diskdb():
    os.system('taskset -c {:d} ./paf_diskdb -a {:s} -b {:s} -c {:s} -d {:s} -e {:s}'.format(diskdb_cpu, diskdb_key, directory, dfname, diskdb_hfname, diskdb_sod))

def process():
    os.system('taskset -c {:d} ./paf_process -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:s} -i {:d} -j {:s} -k {:s} -l {:d}'.format(process_cpu, diskdb_key, process_key, diskdb_ndf, nrun_blk, process_nstream, process_ndf, process_sod, gpu, process_hfname, directory, stream))

def fold():
    # If we only have one visible GPU, we will have to set it to 0;
    if (multi_gpu):
        os.system('dspsr -cpu {:d} -E {:s}.par {:s} -cuda {:d},{:d} -L {:d} -A'.format(fold_cpu, psrname, process_kfname, gpu, gpu, subint))
    else:
        print ('dspsr -cpu {:d} -E {:s}.par {:s} -cuda 0,0 -L {:d} -A'.format(fold_cpu, psrname, process_kfname, subint))
        os.system('dspsr -cpu {:d} -E {:s}.par {:s} -cuda 0,0 -L {:d} -A'.format(fold_cpu, psrname, process_kfname, subint))
         
def main():
    # Create key files
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    diskdb_key_file = open(diskdb_kfname, "w")
    diskdb_key_file.writelines("DADA INFO:\n")
    diskdb_key_file.writelines("key {:s}\n".format(diskdb_key))
    diskdb_key_file.close()

    # Create key files
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    process_key_file = open(process_kfname, "w")
    process_key_file.writelines("DADA INFO:\n")
    process_key_file.writelines("key {:s}\n".format(process_key))
    process_key_file.close()

    os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(diskdb_key, diskdb_rbufsz, diskdb_nbuf, diskdb_nreader))
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(process_key, process_rbufsz, process_nbuf, process_nreader))
        
    t_diskdb  = threading.Thread(target = diskdb)
    t_process = threading.Thread(target = process)
    t_fold    = threading.Thread(target = fold)
    
    t_diskdb.start()
    t_process.start()
    t_fold.start()
    
    t_diskdb.join()
    t_process.join()
    t_fold.join()

    os.system("dada_db -d -k {:s}".format(diskdb_key))
    os.system("dada_db -d -k {:s}".format(process_key))
    
    os.system("mv *.ar {:s}".format(directory))

if __name__ == "__main__":
    main()
