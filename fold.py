#!/usr/bin/env python

# tempo2 -f mypar.par -pred "sitename mjd1 mjd2 freq1 freq2 ntimecoeff nfreqcoeff seg_length"

# I made assumption here:
# 1. scale calculation uses only one buffer block, no matter how big the block is;
# 2. numa node index is 0 and 1, nic ip end with 1 and 2, gpu index is 0 and 1;
# ./fold.py -d 0 -n 1 -c fold.conf -l 3600 -f 2

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

freq = 1340.5  # it should be the value from main startup GUI of TOS plus 0.5

# Read in command line arguments
parser = argparse.ArgumentParser(description='Fold data from file or from BMF stream')
parser.add_argument('-d', '--debug', type=int, nargs='+',
                    help='Flag for debug, 1 fold data from file, 0 fold data from BMF stream')
parser.add_argument('-c', '--cfname', type=str, nargs='+',
                    help='The name of configuration file')
parser.add_argument('-n', '--numa', type=int, nargs='+',
                    help='On which numa node we do the work, 0 or 1')
parser.add_argument('-l', '--length', type=float, nargs='+',
                help='Length of data receiving')
parser.add_argument('-f', '--first_final', type=int, nargs='+',
                    help='First run or final run, 0 for first run and create shared memory, 1 for last run and destroy shared memory, the rest does nothing')

args         = parser.parse_args()
debug        = args.debug[0]
cfname       = args.cfname[0]
numa         = args.numa[0]
length       = args.length[0]
nic          = numa + 1
first_final  = args.first_final[0]

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
process_key        =  ConfigSectionMap("ProcessConf")['key']
process_kfname     =  "{:s}_nic{:d}.key".format(ConfigSectionMap("ProcessConf")['kfname_prefix'], nic)
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
process_rbufsz     = int(0.5 * capture_rbufsz * process_osampratei / process_nchanratei)

process_hfname     = ConfigSectionMap("ProcessConf")['hfname']
process_cpu        = ncpu_numa * numa + capture_ncpu
process_dir        = ConfigSectionMap("ProcessConf")['dir']

# Fold configuration
fold_cpu = ncpu_numa * numa + capture_ncpu + 1
pfname   = ConfigSectionMap("FoldConf")['pfname']
subint   = int(ConfigSectionMap("FoldConf")['subint'])

# Diskdb configuration
diskdb_cpu     = ncpu_numa * numa + capture_ncpu + 2
diskdb_sod     = ConfigSectionMap("DiskdbConf")['sod']
diskdb_hfname  = ConfigSectionMap("DiskdbConf")['hfname']
diskdb_dir     = ConfigSectionMap("DiskdbConf")['dir']
diskdb_dfname  = ConfigSectionMap("DiskdbConf")['dfname']

# Check the buffer block can be covered with multiple run of multiple streams
if (capture_ndf % (process_nstream * process_ndf)):
    print "Multiple run of multiple streams can not cover single ring buffer block, please edit configuration file {:s}".format(cfname)
    exit()
else:
    nrun_blk = capture_ndf / (process_nstream * process_ndf)

def diskdb():
    os.system('taskset -c {:d} ./paf_diskdb -k {:s} -s {:s} -d {:s} -n {:s} -h {:s} -g {:d}'.format(diskdb_cpu, capture_key, diskdb_sod, diskdb_dir, diskdb_dfname, diskdb_hfname, numa))

def capture():
    time.sleep(sleep_time)
    os.system("./paf_capture -k {:s} -l {:f} -n {:d} -h {:s} -f {:f} -e {:s} -s {:s} -r {:d} -d 0".format(capture_key, length, nic, capture_hfname, freq, capture_efname, capture_sod, capture_ndf))

def process():
    time.sleep(0.5 * sleep_time)
    if debug:
        os.system('taskset -c {:d} ./paf_process -i {:s} -o {:s} -c {:d} -d {:d} -s {:s} -h {:s} -n {:d} -p {:d} -f {:s} -b {:d} -g 1'.format(process_cpu, capture_key, process_key, capture_ndf, numa, process_sod, process_hfname, process_nstream, process_ndf, process_dir, nrun_blk))
    else:
        os.system('taskset -c {:d} ./paf_process -i {:s} -o {:s} -c {:d} -d {:d} -s {:s} -h {:s} -n {:d} -p {:d} -f {:s} -b {:d} -g 0'.format(process_cpu, capture_key, process_key, capture_ndf, numa, process_sod, process_hfname, process_nstream, process_ndf, process_dir, nrun_blk))
        
def fold_with_second_ringbuf():
    # Create key files
    # For current version, we only need to create share memory at the first time
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    process_key_file = open(process_kfname, "w")
    process_key_file.writelines("DADA INFO:\n")
    process_key_file.writelines("key {:s}\n".format(process_key))
    process_key_file.close()

    if(first_final == 0):
        os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(process_key, process_rbufsz, process_nbuf, process_nreader))
    os.system('dspsr -cpu {:d} -E {:s} {:s} -cuda {:d},{:d} -L {:d} -A'.format(fold_cpu, pfname, process_kfname, numa, numa, subint))
    if(first_final == 1):
        os.system("dada_db -k {:s} -d".format(process_key))
    
def capture_process_with_first_ringbuf():
    # Create key files
    # For current version, we only need to create share memory at the first time
    # and destroy share memory at the last time
    # this will save prepare time for the pipeline as well
    capture_key_file = open(capture_kfname, "w")
    capture_key_file.writelines("DADA INFO:\n")
    capture_key_file.writelines("key {:s}\n".format(capture_key))
    capture_key_file.close()

    if(first_final == 0):
        os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(capture_key, capture_rbufsz, capture_nbuf, capture_nreader))
    
    # Start threads
    t_process = threading.Thread(target = process)
    t_process.start()
    if debug:
        t_diskdb  = threading.Thread(target = diskdb)
        t_diskdb.start()
    else:
        t_capture = threading.Thread(target = capture)
        t_capture.start()

    # Join threads
    t_process.join()
    if debug:
        t_diskdb.join()
    else:
        t_capture.join()
    if(first_final == 1):
        os.system("dada_db -k {:s} -d".format(capture_key))
        
def main():
    t_first = threading.Thread(target = capture_process_with_first_ringbuf)
    t_second = threading.Thread(target = fold_with_second_ringbuf)
    
    t_first.start()
    t_second.start()
    
    t_first.join()
    t_second.join()

if __name__ == "__main__":
    main()
