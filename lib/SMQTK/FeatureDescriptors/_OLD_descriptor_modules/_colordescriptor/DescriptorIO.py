"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os, struct
import numpy             # you need to have NumPy installed

def __pointsToMatrix(points):
    """Function to parse the point location part of the KOEN1 format"""
    t = []
    for p in points:
        #assert p[0] == "<" and p[-1] == ">"
        parts = p[1:-1].split(" ")
        assert parts[0] == "CIRCLE"

        t.append(parts[1:])

        #print parts, m.shape
        #m[i, 0] = float(parts[1])   # x
        #m[i, 1] = float(parts[2])   # y
        #m[i, 2] = float(parts[3])   # scale
        #m[i, 3] = float(parts[4])   # orientation
        #m[i, 4] = float(parts[5])   # cornerness

    return numpy.matrix(t, dtype=numpy.float64)

def parseKoen1(data, parsePoints=False):
    """Parse the KOEN1 format into two matrices: points and descriptors.

    Data contains the raw bytes of the file."""
    lines = data.splitlines()

    dimensionality = int(lines[1])
    regionCount = int(lines[2])
    points = []
    descriptors = []
    for i in range(regionCount):
        line = lines[i+3]
        parts = line.split(";")

        points.append(parts[0])
        descriptors.append(parts[1].split())
    if parsePoints:
        points = __pointsToMatrix(points)
    return points, numpy.matrix(descriptors, dtype=numpy.float64)

def parseBinaryDescriptors(data):
    """Parse the BINDESC1 format into two matrices: points and descriptors.

    Data contains the raw bytes of the file."""
    header = data[0:32]
    assert header[:8] == "BINDESC1"
    #assert header[8:16] == "CIRCLE  "
    values = struct.unpack("<4I", header[16:])
    elementsPerPoint = values[0]
    dimensionCount = values[1]
    pointCount = values[2]
    bytesPerElement = values[3]

    if bytesPerElement == 8:
        dt = numpy.float64
    elif bytesPerElement == 4:
        dt = numpy.float32
    else:
        raise ValueError("Bytes per element unknown: %d" % bytesPerElement)

    # New way of point and descriptor extraction
    points = numpy.fromstring(data[32:(elementsPerPoint * pointCount * bytesPerElement)+32], dtype=dt)
    descriptors = numpy.fromstring(data[(elementsPerPoint * pointCount * bytesPerElement)+32:], dtype=dt)

    # Reshape the arrays into matrices
    points = numpy.reshape(points, (pointCount, elementsPerPoint))
    descriptors = numpy.reshape(descriptors, (pointCount, dimensionCount))

    return points, descriptors

def readDescriptors(filename):
    """Load descriptors from filename or file descriptor.

    Identification of KOEN1/BINDESC1 format is automatic. Returns two matrices:
    the first one contains the points with a typical size of (n,5) and
    descriptors with a typical size of (n,d) with d the dimensionality of
    the descriptor."""
    if hasattr(filename, "read"):
        f = filename
    else:
        f = open(filename, "rb")
    identify = f.read(4)
    f.seek(-4, os.SEEK_CUR)

    # text file?
    if identify == "KOEN":
        return parseKoen1(f.read(), True)

    # this is a binary file
    header = f.read(32)
    assert header[:8] == "BINDESC1"
    #assert header[8:16] == "CIRCLE  "
    values = struct.unpack("<4I", header[16:])
    elementsPerPoint = values[0]
    dimensionCount = values[1]
    pointCount = values[2]
    bytesPerElement = values[3]
    if bytesPerElement == 8:
        dt = numpy.float64
    elif bytesPerElement == 4:
        dt = numpy.float32
    else:
        raise ValueError("Bytes per element unknown: %d" % bytesPerElement)
    points = numpy.fromstring(f.read(elementsPerPoint * pointCount * bytesPerElement), dtype=dt)
    descriptors = numpy.fromstring(f.read(dimensionCount * pointCount * bytesPerElement), dtype=dt)
    points = numpy.reshape(points, (pointCount, elementsPerPoint))
    descriptors = numpy.reshape(descriptors, (pointCount, dimensionCount))
    if not hasattr(filename, "read"):
        f.close()
    return points, descriptors

def writeBinaryDescriptors(filename, points, descriptors, info=""):
    """Write the BINDESC1 format from two matrices: points and descriptors."""

    elementsPerPoint = points.shape[1]
    dimensionCount = descriptors.shape[1]
    pointCount = descriptors.shape[0]
    bytesPerElement = 8

    if pointCount != points.shape[0]:
        raise ValueError("Shape mismatch: should have same number of rows")

    if hasattr(filename, "write"):
        f = filename
    else:
        f = open(filename, "wb")
    header = "BINDESC1" + (info + " " * 8)[:8]
    header += struct.pack("<4I", elementsPerPoint, dimensionCount, pointCount, bytesPerElement)
    f.write(header)

    if bytesPerElement == 8:
        dt = numpy.float64
    elif bytesPerElement == 4:
        dt = numpy.float32
    else:
        raise ValueError("Bytes per element unknown: %d" % bytesPerElement)

    # New way of point and descriptor extraction
    data = points.astype(dt).tostring()
    f.write(data)
    data = descriptors.astype(dt).tostring()
    f.write(data)
    f.close()

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    print "Loading", filename
    points, descriptors = readDescriptors(filename)
    print "Points in the file:", points.shape
    print points
    print "Descriptors in the file:", descriptors.shape
    print descriptors
