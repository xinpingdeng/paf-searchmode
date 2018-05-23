#!/usr/bin/env python

import os

dname = "searchmode"

os.system("docker exec -it {:s} /bin/bash".format(dname))
