#!/usr/bin/env python

import os
import argparse

parser = argparse.ArgumentParser(description='Rebuild the software under the same directory with and without debug output')
parser.add_argument('debug', metavar='d', type=int, nargs='+',
                    help='Flag for debug, 1 build the software with debug output, 0 build the software without debug output')

args = parser.parse_args()
debug = args.debug[0]

#os.system("ipcrm -a")
os.system("make clean")
os.system("make DEBUG={:d}".format(debug))
os.system("make clean")
