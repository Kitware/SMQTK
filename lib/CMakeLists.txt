# Installation
install(DIRECTORY EventContentDescriptor
  DESTINATION "${python_site_packages}"
  )

install(DIRECTORY SMQTK_Backend
  DESTINATION "${python_site_packages}"
  USE_SOURCE_PERMISSIONS
  )

install(DIRECTORY Utils
  DESTINATION "${python_site_packages}"
  )

install(PROGRAMS SMQTK_Backend/FeatureManagerServer.py
  DESTINATION bin
  RENAME FeatureManagerServer
  )
