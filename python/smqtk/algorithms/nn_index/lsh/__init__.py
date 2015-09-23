"""
This module contains various implementations of locality-sensitive-hash-based
indexing techniques.
"""

from .itq import ITQNearestNeighborsIndex


__author__ = 'purg'


NN_INDEX_CLASS = [
    ITQNearestNeighborsIndex
]
