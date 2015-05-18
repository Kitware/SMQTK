#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

rootPath = os.environ['KWDEMO_KEY']

cfg = """
[global]
server.socket_port: 8080
tools.proxy.on: True

[database]
uri: "mongodb://%HOSTIP%:27017/%ROOTPATH%"

[server]
# Set to "production" or "development"
mode: "production"
api_root: "api/v1"
static_root: "static"
api_static_root: "../static"
"""

hostip = os.popen(
    "netstat -nr | grep '^0\.0\.0\.0' | awk '{print $2}'").read().strip()
cfg = cfg.replace('%HOSTIP%', hostip)
cfg = cfg.replace('%ROOTPATH%', rootPath)
print cfg.strip()