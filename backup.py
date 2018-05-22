#!/usr/bin/env python

import datetime
import os

os.system('rm *~')

now = datetime.datetime.now()

top_dir = now.strftime("%Y%m%d")
if not os.path.isdir(top_dir):
    os.system('mkdir {:s}'.format(top_dir))
os.chdir(top_dir)

low_dir = now.strftime("%H")
if not os.path.isdir(low_dir):
    os.system('mkdir {:s}'.format(low_dir))
os.system('cp ../* {:s}'.format(low_dir))
