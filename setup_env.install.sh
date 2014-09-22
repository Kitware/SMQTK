#
# Setup the run environment (install environment)
#

# Assuming bash environment
export SMQTK_INSTALL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SMQTK_INSTALL}/setup_backend.sh"
# System PATH and PYTHONPATH the same as the Backend's
export SMQTK_SYSTEM_SETUP=1
