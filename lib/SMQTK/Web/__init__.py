"""

SMQTK Web Applications

"""

# Convenience imports of Flask web application classes
from .SMQTKSearchApp.base_app import SMQTKSearchApp

APPLICATIONS = [
    SMQTKSearchApp,
]
