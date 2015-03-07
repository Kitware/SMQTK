"""

SMQTK Web Applications

"""

# Convenience imports of Flask web application classes
from .SMQTKSearchApp.app import SMQTKSearchApp

APPLICATIONS = [
    SMQTKSearchApp,
]
