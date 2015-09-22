# Installation
install(DIRECTORY EventContentDescriptor
  DESTINATION "${python_site_packages}"
  )

install(DIRECTORY smqtk
  DESTINATION "${python_site_packages}"
  USE_SOURCE_PERMISSIONS
  )

install(DIRECTORY Utils
  DESTINATION "${python_site_packages}"
  )
