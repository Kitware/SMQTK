from smqtk.utils import Configurable, SmqtkObject


#  noinspection PyAbstractClass
class SmqtkRepresentation (SmqtkObject, Configurable):
    """
    Interface for data representation interfaces and implementations.

    Data should be serializable, so this interface adds abstract methods for
    serializing and de-serializing SMQTK data representation instances.

    """
    # TODO(paul.tunison): Add serialization abstract method signatures here.
    # - Could start with just requiring implementing sub-classes to
    #   ``__getstate__`` and ``__setstate__`` methods required for pickle
    #   interface.
