#!/usr/bin/env python

import os

dir_name = "/beegfs/DENG/docker"
fil_fname = 'I_58227.55061683_beam_12_8bit.fil'
dada_fname = "2018-04-17-19:22:11_0000000000000000.000000.dada"
out_fname = "new.fil"

out_file = open("{:s}/{:s}".format(dir_name, out_fname), "w")
dada_file = open("{:s}/{:s}".format(dir_name, dada_fname), "r")
fil_file = open("{:s}/{:s}".format(dir_name, fil_fname), "r")

out_file.write(fil_file.read(342))
dada_file.seek(4096)
out_file.write(dada_file.read())

out_file.close()
dada_file.close()
fil_file.close()
