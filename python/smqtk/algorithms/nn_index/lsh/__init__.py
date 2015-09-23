"""
This module contains various implementations of locality-sensitive-hash-based
indexing techniques.
"""
__author__ = 'purg'

from .itq import ITQNearestNeighborsIndex


SIMILARITY_INDEX_CLASS = [
    ITQNearestNeighborsIndex
]
