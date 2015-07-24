import abc
import os

from smqtk.utils.plugin import get_plugins

class DetectAndDescribe (object):
	def __init__(self):
		pass

	@abc.abstractmethod
	def detect_and_describe(self, data):
		"""
		This method is to be implemented in classes that implement DetectAndDescribe. It should do what the name
		implies, given input data -- 
			1) sample the points (i.e., detect)
			2) take the sampled points and compute the implemented descriptor (i.e., describe)

		Of course, an implementation of DetectAndDescribe does not have to have seperate detect and describe
		stages, as would be the case for some implementataions such as colorDescriptor.
		"""
		pass


def get_detect_and_describe():
	"""
    Discover and return all DetectAndDescribe classes found in the given plugin
    search directory. Keys in the returned map are the names of the discovered
    classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``CONTENT_DESCRIPTOR_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :return: Map of discovered class object of type ``ContentDescriptor`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
	this_dir = os.path.abspath(os.path.dirname(__file__))
	helper_var = "DETECT_AND_DESCRIBE_CLASS"
	return get_plugins(__name__, this_dir, helper_var, DetectAndDescribe,
		lambda c: c.is_usable())