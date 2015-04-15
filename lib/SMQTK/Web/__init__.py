"""

SMQTK Web Applications

"""

# Convenience imports of Flask web application classes
from .SMQTKSearchApp.base_app import SMQTKSearchApp
from .DescriptorService.server import DescriptorServiceServer

APPLICATIONS = [
    SMQTKSearchApp,
    DescriptorServiceServer
]
