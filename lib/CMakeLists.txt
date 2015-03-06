# SMQTK python default configuration file
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/smqtk_config.build.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/smqtk_config.py"
               @ONLY
               )
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/smqtk_config.install.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/smqtk_config.install.py"
               @ONLY
               )

# Installation
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/smqtk_config.install.py"
  DESTINATION "${python_site_packages}"
  RENAME "smqtk_config.py"
  )

install(DIRECTORY EventContentDescriptor
  DESTINATION "${python_site_packages}"
  )

install(DIRECTORY SMQTK
  DESTINATION "${python_site_packages}"
  USE_SOURCE_PERMISSIONS
  )

install(DIRECTORY Utils
  DESTINATION "${python_site_packages}"
  )

#install(PROGRAMS SMQTK/FeatureManagerServer.py
#  DESTINATION bin
#  RENAME FeatureManagerServer
#  )
