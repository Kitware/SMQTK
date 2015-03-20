"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import csv

"""Video Properties using FFMPEG pipes."""

import os
import subprocess
import re
from optparse import OptionParser

# SET this variable to your system ffmpeg path
#FFMPEG_PATH = 'C:\\ProgramFiles\\ffmpeg\\ffmpeg-git-f514695-win64-static\\bin\\ffmpeg.exe'
FFMPEG_PATH = 'ffmpeg'

def video_properties(in_video, verbose=False):
    """
    Finds video property information using FFMPEG
    @param in_video: path to the video file
    @return: [ [width, height], duration, fps], in units of [ [pixels, pixels], seconds, Hz]
    """
    cmd = ' '.join([FFMPEG_PATH, '-i', str(in_video)])
    print cmd

    ffmpeg_props_log = "ffmpeg_props.log"
    rc = os.system (cmd + " > " + ffmpeg_props_log + " 2>&1")
    lf = open(ffmpeg_props_log, 'r')
    info = lf.read()
    lf.close()
    if verbose:
        print "info: [" + info + "]"

    # find output image size
    m = None
    m = re.search("Stream.*Video:.* (\d+)x(\d+)", info)
    if m:
        width  = int(m.group(1))
        height = int(m.group(2))
    else:
        width  = None
        height = None

    # find video duration
    m = None
    m = re.search("Duration:\s* (\d+):(\d+):(\d+).(\d*)", info)
    if m:
        duration = 3600*float(m.group(1)) + 60*float(m.group(2)) + float(m.group(3))
        if m.group(4) != '' :
            duration += float(''.join(['0.',m.group(4)]))
    else:
        duration = None

    # find video fps
    m = None
    m = re.search(".(\d*).(\d+).fps", info)
    if m:
        if m.group(1)=='':
            fps = float(m.group(2))
        else:
            fps = float(m.group(1)) + float(''.join(['0.',m.group(2)]))
    else:
        fps = None

    return [ [width, height], duration, fps]


def parse_video_properties(filename):
    """
    Parse video properties file in the following CSV format
    Col 0: HVC%d, which is video clipid with prefix HVC
    Col 1: width
    Col 2: height
    Col 3: duration
    Col 4: fps
    @param filename: file with video properties
    @return: dictionary indexed by integer clipid key, pointing to the contents with [w,h,duration,fps]
    """

    properties = dict()

    with open(filename) as fin:
        for line in csv.reader(fin):
            clipid = int(os.path.basename(line[0])[3:])
            width  = int(line[1])
            height = int(line[2])
            duration = float(line[3])
            fps      = float(line[4])

            properties[clipid] = [width, height, duration, fps]

    return properties


def main():
    usage = "usage: %prog in_videos out_file\n\n"
    usage += "  Record video properties, and store them\n"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    file_in = args[0]
    file_out = open(args[1], 'w')

    in_videos = sorted([f.strip() for f in open(file_in,'r') if f.strip()])
    n = len(in_videos)

    for i in range(n):
        i_file = in_videos[i]
        [[w,h], duration, fps] = video_properties(i_file)
        i_video =os.path.splitext(os.path.basename(i_file))[0]
        #i_video = os.path.join(os.path.basename(os.path.dirname(i_file)),
        #                       os.path.splitext(os.path.basename(i_file))[0])

        file_out.write('%s, %d, %d, %0.2f, %0.2f\n'%(i_video, w, h, duration, fps))

    file_out.close()


if __name__ == "__main__":
    main()

