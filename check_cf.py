#!/usr/bin/env python

import socket, struct
import numpy as np

ip    = "10.17.8.1"
ports = [17102, 17103]

for port in ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.bind((ip, port))
    pkt, addr = sock.recvfrom(1<<16)

    data = np.array(np.fromstring(pkt, dtype='uint64'))
    print bin(data[0])
    print bin(struct.unpack("<Q", struct.pack(">Q", data[0]))[0])
    print bin(data[1])
    print bin(struct.unpack("<Q", struct.pack(">Q", data[1]))[0])
    print bin(data[2])
    print bin(struct.unpack("<Q", struct.pack(">Q", data[2]))[0])
    
    
    sock.close()

#    1110 00000000 01001011 00000101 00001101 00000000 00000110 00000000
