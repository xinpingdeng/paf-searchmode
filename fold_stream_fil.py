#!/usr/bin/env python

# ./fold_stream_fil.py -a fold_stream_fil.conf -b 0 -c 10 -d /beegfs/DENG/docker/ -e J0218+4232 -f 1
# tempo2 -f mypar.par -pred "sitename mjd1 mjd2 freq1 freq2 ntimecoeff nfreqcoeff seg_length"

# I made assumption here:
# 1. scale calculation uses only one buffer block, no matter how big the block is;
# 2. numa node index is 0 and 1, nic ip end with 1 and 2, gpu index is 0 and 1;

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
hdr    = 0       # For fold mode, we do not capture header of each packer;
stream = 1       # For real-time folding, we fold on data stream

# Read in command line arguments
parser = argparse.ArgumentParser(description='Fold data from BMF stream')
parser.add_argument('-a', '--cfname', type=str, nargs='+',
                    help='The name of configuration file')
parser.add_argument('-b', '--numa', type=int, nargs='+',
                    help='On which numa node we do the work, 0 or 1')
parser.add_argument('-c', '--length', type=float, nargs='+',
                help='Length of data receiving')
parser.add_argument('-d', '--directory', type=str, nargs='+',
                    help='In which directory we record the data and read configuration files and parameter files')
parser.add_argument('-e', '--psrname', type=str, nargs='+',
                    help='The name of pulsar')
parser.add_argument('-f', '--visiblegpu', type=str, nargs='+',
                    help='Visible GPU, the parameter is for the usage inside docker container.')

args         = parser.parse_args()
cfname       = args.cfname[0]
numa         = args.numa[0]
length       = args.length[0]
nic          = numa + 1
directory    = args.directory[0]
psrname      = args.psrname[0]
if(args.visiblegpu[0]==''):
    multi_gpu = 1;
elif(args.visiblegpu[0]=='all'):
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
sleep_time   = int(ConfigSectionMap("BasicConf")['sleep_time'])
ncpu_numa    = int(ConfigSectionMap("BasicConf")['ncpu_numa'])

# Capture configuration
capture_ncpu    = int(ConfigSectionMap("CaptureConf")['ncpu'])
capture_ndf 	= int(ConfigSectionMap("CaptureConf")['ndf'])
capture_nbuf    = ConfigSectionMap("CaptureConf")['nblk']
capture_key     = ConfigSectionMap("CaptureConf")['key']
capture_key     = format(int("0x{:s}".format(capture_key), 0) + 2 * nic, 'x')
capture_kfname  = "{:s}_nic{:d}.key".format(ConfigSectionMap("CaptureConf")['kfname_prefix'], nic)
capture_efname  = ConfigSectionMap("CaptureConf")['efname']
capture_hfname  = ConfigSectionMap("CaptureConf")['hfname']
capture_nreader = ConfigSectionMap("CaptureConf")['nreader']
capture_sod     = ConfigSectionMap("CaptureConf")['sod']
capture_rbufsz  = capture_ndf *  nchk_nic * 7168

# Process configuration
process_key        = ConfigSectionMap("ProcessConf")['key']
process_kfname     = "{:s}_nic{:d}.key".format(ConfigSectionMap("ProcessConf")['kfname_prefix'], nic)
process_key        = format(int("0x{:s}".format(process_key), 0) + 2 * nic, 'x')
process_sod        = ConfigSectionMap("ProcessConf")['sod']
process_nreader    = ConfigSectionMap("ProcessConf")['nreader']
process_nbuf       = ConfigSectionMap("ProcessConf")['nblk']
process_ndf        = int(ConfigSectionMap("ProcessConf")['stream_ndfstp'])
process_nstream    = int(ConfigSectionMap("ProcessConf")['nstream'])
#process_byte       = int(ConfigSectionMap("ProcessConf")['nbyte_out'])
#process_nchanfinal = int(ConfigSectionMap("ProcessConf")['nchan_final'])
#process_rbufsz     = nsamp_df * process_nstream * process_ndf * npol_samp * ndim_pol * process_nchanfinal * process_byte
process_osampratei = float(ConfigSectionMap("ProcessConf")['osamp_ratei'])
process_nchanratei = float(ConfigSectionMap("ProcessConf")['nchan_ratei'])
process_nchanint   = float(ConfigSectionMap("ProcessConf")['nchan_int'])
#process_rbufsz     = int(0.5 * capture_rbufsz * process_osampratei / process_nchanratei)
process_rbufsz     = int(0.5 * capture_rbufsz * process_osampratei / process_nchanratei / process_nchanint / 4)

process_hfname     = ConfigSectionMap("ProcessConf")['hfname']
process_cpu        = ncpu_numa * numa + capture_ncpu

# Fold configuration
fold_cpu = ncpu_numa * numa + capture_ncpu + 1
subint   = int(ConfigSectionMap("FoldConf")['subint'])

# Check the buffer block can be covered with multiple run of multiple streams
if (capture_ndf % (process_nstream * process_ndf)):
    print "Multiple run of multiple streams can not cover single ring buffer block, please edit configuration file {:s}".format(cfname)
    exit()
else:
    nrun_blk = capture_ndf / (process_nstream * process_ndf)

def capture():
    time.sleep(sleep_time)
    os.system("./paf_capture -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:s} -g {:s} -i {:f} -j {:f} -k {:s}".format(capture_key, capture_sod, capture_ndf, hdr, nic, capture_hfname, capture_efname, freq, length, directory))

def process():
    time.sleep(0.5 * sleep_time)
    os.system('taskset -c {:d} ./paf_process -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:s} -i {:d} -j {:s} -k {:s} -l {:d}'.format(process_cpu, capture_key, process_key, capture_ndf, nrun_blk, process_nstream, process_ndf, process_sod, numa, process_hfname, directory, stream))

def fold():
    # If we only have one visible GPU, we will have to set it to 0;
    if (multi_gpu):
        #os.system('dspsr -cpu {:d} -E {:s}.par {:s} -cuda {:d},{:d} -L {:d} -A'.format(fold_cpu, psrname, process_kfname, numa, numa, subint))
        os.system('dspsr -cpu {:d} -E {:s}.par {:s} -L {:d} -A'.format(fold_cpu, psrname, process_kfname, subint))
    else:
        #os.system('dspsr -cpu {:d} -E {:s}.par {:s} -cuda 0,0 -L {:d} -A'.format(fold_cpu, psrname, process_kfname, subint))
        os.system('dspsr -cpu {:d} -E {:s}.par {:s} -L {:d} -A'.format(fold_cpu, psrname, process_kfname, subint))   
         
def main():
    # Create key files
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    capture_key_file = open(capture_kfname, "w")
    capture_key_file.writelines("DADA INFO:\n")
    capture_key_file.writelines("key {:s}\n".format(capture_key))
    capture_key_file.close()

    # Create key files
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    process_key_file = open(process_kfname, "w")
    process_key_file.writelines("DADA INFO:\n")
    process_key_file.writelines("key {:s}\n".format(process_key))
    process_key_file.close()

    os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(capture_key, capture_rbufsz, capture_nbuf, capture_nreader))
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(process_key, process_rbufsz, process_nbuf, process_nreader))
        
    t_capture = threading.Thread(target = capture)
    t_process = threading.Thread(target = process)
    t_fold    = threading.Thread(target = fold)
    
    t_capture.start()
    t_process.start()
    t_fold.start()
    
    t_capture.join()
    t_process.join()
    t_fold.join()

    os.system("dada_db -d -k {:s}".format(capture_key))
    os.system("dada_db -d -k {:s}".format(process_key))
    
    os.system("mv *.ar {:s}".format(directory))

if __name__ == "__main__":
    main()
