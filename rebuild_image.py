#!/usr/bin/env python

import os

image_name = "searchmode"
user_name  = "xinpingdeng"

os.system("docker build -t {:s} .".format(image_name))
os.system("docker tag {:s} {:s}/{:s}".format(image_name, user_name, image_name))
os.system("docker push {:s}/{:s}".format(user_name, image_name))
