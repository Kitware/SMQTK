"""

SMQTK Web Applications

"""

# Convenience imports of Flask web application classes
from .search_app.base_app import SMQTKSearchApp
from .descriptor_service.server import DescriptorServiceServer

APPLICATIONS = [
    SMQTKSearchApp,
    DescriptorServiceServer
]
