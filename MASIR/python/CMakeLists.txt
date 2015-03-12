
###
# Configuration
#

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/masir_config.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/masir_config.py"
               @ONLY
               )


###
# Installation
#
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/masir_config.py"
  DESTINATION "${python_site_packages}"
  RENAME masir_config.py
  )

install(
  DIRECTORY masir
  DESTINATION "${python_site_packages}"
  )
