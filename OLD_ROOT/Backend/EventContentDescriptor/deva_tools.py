"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
"""
Collection of tools for DEVA-related parsing/writing
"""

import csv
import numpy as np

##############################
# Reference file
##############################

def parse_reference_file_3cols(filename_ref):
    """
    Read a DEVA-format Reference file, and outputs matrix with the following format
    - col_0: clip ids,
    - col_1: target labels
    - col_2: yes (1) or no (0).

    @param filename_ref: full path to the reference file
    @return: list of 3D tuples (clipid, target index, 1 or 0), can be easily converted to np array later
    """

    fin_ = open(filename_ref, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    lines = lines[1::] # get rid of the first line header (perhaps better way to do this?)

    parsings = []
    for line in lines:
        clipid = int(line[0][0:6])
        eid = int(line[0][-3::])
        decision = 0
        if line[1] == 'y':
            decision = 1
        parsings.append([clipid, eid, decision])

    return parsings

def parse_reference_file_2cols(filename_ref):
    """
    Read a DEVA-format Reference file, and outputs matrix with the following format
    - col_0: clip ids,
    - col_1: gt eid (assume each clip has either single eid (>0), or None (0)

    @param filename_ref: full path to the reference file
    @return: 2D list, can be easily converted to np array later
    """

    cols3 = parse_reference_file_3cols(filename_ref)
    cols2 = dict()

    for [clipid, eid, decision] in cols3:
        eid_gt = 0
        if decision == 1:
            eid_gt = eid

        try:
            val = cols2[clipid] # if it does not exist, error throw.
            if (val == 0) and (eid_gt > 0):
                cols2[clipid] = eid_gt # udpate
        except:
            cols2[clipid] = eid_gt

    return [ [clipid, eid_gt] for (clipid, eid_gt) in cols2.iteritems() ]


##############################
# Honeywell format
##############################

def parse_HW_split_files(filename):

    # simple hacky code
    fin_ = open(filename, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    parsings = []
    for line in fin:
        clipid = int(line[0][3:])
        desc = line[1].strip()
        eid = None
        if desc == 'None':
            eid = 0
        else:
            eid = int(desc[-3:])
        decision = None
        if (len(desc) > 4) or desc==None:
            decision = 0
        else:
            decision = 1
        parsings.append([clipid, eid, decision])

    return parsings



##############################
# System File Parsing
##############################

def parse_system_file(filename):
    """
    Read a DEVA-format System File.
    Returns a list with the following four elements
    - 0: event id
    - 1: detection threshold
    - 2: DetectionTPT
    - 3: EAGTPT

    @param filename: full path to the system file
    @return: list of list, each element list has four items from each row of system file
    """

    fin_ = open(filename, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    lines = lines[1::] # get rid of the first line header (perhaps better way to do this?)

    parsings = []
    for line in lines:
        eid = int(line[0][1:])
        threshold = float(line[1])
        detection_tpt = float(line[2])
        eag_tpt = float(line[3])
        parsings.append([eid, threshold, detection_tpt, eag_tpt])

    return parsings



##############################
# Judgement File Parsing
##############################

def parse_groundtruth_from_judgementMD(filename,
                                       target_eid = None, export_numpy_format = False,
                                       val_positive = 1,
                                       val_near_miss = 0.5,
                                       val_related = 0.2,
                                       val_not_sure = 0.1,
                                       val_NULL_NULL = 1,
                                       val_NULL = 0):
    """
    Parse target ground truth and judgements
    Each item in the returned outputs is a list of ['%06d' clipid, 'E%03d' target eid, and judgement string]

    If export_numpy_format flag is True, then,
    return n-by-3 array where:
    col0 = clipid,
    col1 = target eid, (NULL class is 0)
    col2 = positive/1 or nearmiss=0.5, related = 0.2, not_sure = 0.1, val_NULL_NULL (1), val_NULL (0)

    @param filename: input file name
    @param target_eid: (integer) if only wants items related to specified target eid, e.g., 5 or 6.
    @param export_numpy_format: return results in numpy format
    @param val_positive: override default numeric value for positive clips
    @param val_near_miss: override default numeric value for near_miss clips
    @param val_related: override default numeric value for related clips
    """
    outputs = []

    with open(filename, 'rb') as fin:
        for line in csv.reader(fin, delimiter = ','):
            if line[0] == 'ClipID':
                continue # ignore header line
            else:
                outputs.append(line[0:3])

    # only keep the outputs that have the desired target eid match
    if target_eid is not None:
        target_str = 'E%03d'%target_eid
        outputs = [output for output in outputs if output[1]==target_str]

    if export_numpy_format:
        outputs_new = []
        for output in outputs:
            clipid = int(output[0])
            eid = None
            if output[1].strip() == 'NULL':
                eid = 0
            else:
                eid = int(output[1][1:])
            judgement = None
            if output[2] == 'positive':
                judgement = val_positive
            elif output[2] == 'near_miss':
                judgement = val_near_miss
            elif output[2] == 'related':
                judgement = val_related
            elif output[2] == 'not_sure':
                judgement = val_not_sure
            elif eid == 0 and output[2]=='NULL':
                judgement = val_NULL_NULL
            elif output[2] == 'NULL':
                judgement = val_NULL
            else:
                continue

            outputs_new.append([clipid, eid, judgement])

        outputs = np.array(outputs_new)
        outputs_new = None

    return outputs


##############################
# Detection file
##############################

def parse_deva_detection_file(file_in):
    """
    Read a DEVA detection file, and outputs a CLIPSx3 matrix
    where
    Col 1 : clipid
    Col 2 : target event
    Col 3 : score
    """
    fin_ = open(file_in, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    lines = lines[1::] # get rid of the first line header (perhaps better way to do this?)

    mat = np.zeros([len(lines), 3])
    count = 0
    for line in lines:
        mat[count][0] = int(line[0][0:-5])
        mat[count][1] = int(line[0][-3::])
        mat[count][2] = float(line[1])
        count += 1
    return mat

def write_deva_detection_file(clipids, scores, eids, fileout, write_mode='w'):
    """
    Given all score information, write deva detection file
    @param clipids: list of clipids (integer number)
    @param scores: list of scores between zero and one
    @param eids: list of event ids corresponding to each trial ID
    @param write_mode: if equal to 'w' then, write a new file, if 'a', then,  append (standard Python mode)
    """
    fout = open(fileout, write_mode)

    tuples = zip(clipids, eids, scores)

    if write_mode == 'w':
        fout.write('"TrialID","Score"\n')

    for (clipid, eid, score) in tuples:
        lineid = "%06d.E%03d"%(clipid, eid)
        fout.write('"%s","%g"\n'%(lineid, score))

    fout.close()


def write_deva_detection_file2(data, fileout, write_mode='w'):
    """
    Given all score information, write deva detection file
    @param data: 2D numpy.array with each row being (clipid, eid, score)
    @param fileout: output filename
    @param write_mode: if equal to 'w' then, write a new file, if 'a', then,  append (standard Python mode)
    """
    fout = open(fileout, write_mode)
    if write_mode == 'w':
        fout.write('"TrialID","Score"\n')

    for _row in data:
        clipid = int(_row[0])
        eid = int(_row[1])
        score = float(_row[2])
        lineid = "%06d.E%03d"%(clipid, eid)
        fout.write('"%s","%g"\n'%(lineid, score))
    fout.close()##############################
# Threshold file
##############################

def parse_deva_threshold_file(file_in):
    """
    Read a DEVA threshold file, and outputs a EventIDx2 matrix
    where
    Col 1 : target event
    Col 2 : threshold
    """
    fin_ = open(file_in, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    lines = lines[1::] # get rid of the first line header (perhaps better way to do this?)

    mat = np.zeros([len(lines), 3])
    count = 0
    for line in lines:
        mat[count][0] = int(line[0][2:5])
        mat[count][1] = float(line[1])
        count += 1
    return mat




###############################
# Scripts below
###############################

def script_create_clipid_eid_file(file_gt, file_out):
    pairs = parse_reference_file_2cols(file_gt)
    with open(file_out, 'wb') as fout:
        for (clipid, eid) in pairs:
            fout.write('%06d, %d\n'%(clipid, eid))


# main
if __name__ == "__main__":
    import sys
    script_create_clipid_eid_file(sys.argv[1], sys.argv[2])



