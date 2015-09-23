"""
This module contains various implementations of locality-sensitive-hash-based
indexing techniques.
"""
__author__ = 'purg'

from .itq import ITQSimilarityIndex


SIMILARITY_INDEX_CLASS = [
    ITQSimilarityIndex
]
