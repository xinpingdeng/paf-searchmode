#!/usr/bin/env python

import os

container_name = "searchmode"
user_name  = "xinpingdeng"
image_name = container_name

os.system("docker commit {:s} {:s}/{:s}".format(container_name, user_name, image_name))
os.system("docker push {:s}/{:s}".format(user_name, image_name))
