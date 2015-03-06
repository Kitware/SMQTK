

#+
# Variables
set(TPL_BUILD_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/TPL-build")
set(TPL_LOCAL_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/TPL-install")
set(TPL_PYTHON_SP "${TPL_LOCAL_INSTALL_PREFIX}/${python_site_packages}")
#-


#+
# Installation of non-TPL things
set(BACKEND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(BACKEND_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/setup_env.build.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/setup_env.build.sh"
  @ONLY
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/setup_env.install.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/setup_env.install.sh"
  @ONLY
  )
#-


# Pre-compiled binary files
add_subdirectory(bin)

# Additional non-standard third-party things
add_subdirectory(TPL)


#+
# Installation
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/setup_env.install.sh"
  DESTINATION .
  RENAME setup_backend.sh
  )

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
#-
