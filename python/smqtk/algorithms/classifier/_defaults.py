from smqtk.representation import ClassificationElementFactory
from smqtk.representation.classification_element.memory import \
    MemoryClassificationElement


# Default classifier element factory for interfaces.
DFLT_CLASSIFIER_FACTORY = ClassificationElementFactory(
    MemoryClassificationElement, {}
)
