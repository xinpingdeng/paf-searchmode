#!/usr/bin/env python

# I made assumption here:
# 1. scale calculation uses only one buffer block, no matter how big the block is;
# 2. nic ip end with 1 and 2, numa node is 0 and 1;
# ./baseband_dump.py -c baseband_dump.conf -n 1 -l 27 -d 1

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

#freq              = 1308.0 # should from metadata

## Get center frequency from multi cast
#multicast_group = '224.1.1.1'
#server_address = ('', 5007)
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create the socket
#sock.bind(server_address) # Bind to the server address
#group = socket.inet_aton(multicast_group)
#mreq = struct.pack('4sL', group, socket.INADDR_ANY)
#sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)  # Tell the operating system to add the socket to the multicast group on all interfaces.
#pkt, address = sock.recvfrom(1<<16)
#data = json.loads(pkt)#['beams_direction']#['beam01']
#freq = float(data['sky_frequency'])

freq = 1340.5

# Read in command line arguments
parser = argparse.ArgumentParser(description='Dump baseband data to files')
parser.add_argument('-c', '--cfname', type=str, nargs='+',
                    help='The name of configuration file')
parser.add_argument('-n', '--numa', type=int, nargs='+',
                    help='On which numa node we do the work, 0 or 1')
parser.add_argument('-d', '--hdr', type=int, nargs='+',
                    help='Record header of packets or not')
parser.add_argument('-l', '--length', type=float, nargs='+',
                help='Length of data receiving')

args   = parser.parse_args()
cfname = args.cfname[0]
numa   = args.numa[0]
length = args.length[0]
hdr    = args.hdr[0]
nic    = numa + 1

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
capture_hdr  	= hdr
capture_ndf 	= int(ConfigSectionMap("CaptureConf")['ndf'])
capture_nbuf    = ConfigSectionMap("CaptureConf")['nblk']
capture_key     = ConfigSectionMap("CaptureConf")['key']
capture_key     = format(int("0x{:s}".format(capture_key), 0) + 2 * nic, 'x')
capture_kfname  = "{:s}_nic{:d}.key".format(ConfigSectionMap("CaptureConf")['kfname_prefix'], nic)
capture_efname  = ConfigSectionMap("CaptureConf")['efname']
capture_nreader = ConfigSectionMap("CaptureConf")['nreader']
capture_sod     = ConfigSectionMap("CaptureConf")['sod']
capture_ncpu    = int(ConfigSectionMap("CaptureConf")['ncpu'])

if int(capture_hdr) == 1:
    capture_hfname  = ConfigSectionMap("CaptureConf")['hfname_hdr']
    capture_rbufsz = capture_ndf * nchk_nic * 7232
else:
    capture_hfname  = ConfigSectionMap("CaptureConf")['hfname']
    capture_rbufsz = capture_ndf * nchk_nic * 7168

# Dbdisk configuration
dbdisk_cpu     = ncpu_numa * numa + capture_ncpu
dbdisk_dir     = ConfigSectionMap("DbdiskConf")['dir']

def capture():
    time.sleep(sleep_time)
    os.system("./paf_capture -k {:s} -l {:f} -n {:d} -h {:s} -f {:f} -e {:s} -s {:s} -r {:d} -d {:d}".format(capture_key, length, nic, capture_hfname, freq, capture_efname, capture_sod, capture_ndf, capture_hdr))

def dbdisk():
    os.system('dada_dbdisk -b {:d} -k {:s} -D {:s} -W'.format(dbdisk_cpu, capture_key, dbdisk_dir))
    
def main():
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:s} -r {:s}".format(capture_key, capture_rbufsz, capture_nbuf, capture_nreader))
    
    t_capture = threading.Thread(target = capture)
    t_dbdisk  = threading.Thread(target = dbdisk)

    t_capture.start()
    t_dbdisk.start()

    t_capture.join()
    t_dbdisk.join()
        
    os.system("dada_db -k {:s} -d".format(capture_key))
    
if __name__ == "__main__":
    main()
