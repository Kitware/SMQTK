#!/bin/bash
mongoimport -d ist -c ads --type csv istdata.txt --fieldFile istfields.txt
