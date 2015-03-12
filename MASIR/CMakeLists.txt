project(MASIR)
cmake_minimum_required(VERSION 2.8)


###
# Versioning
#
set(MASIR_VERSION_MAJOR 0)
set(MASIR_VERSION_MINOR 1)
set(MASIR_VERSION_PATCH 0)
set(MASIR_VERSION "${MASIR_VERSION_MAJOR}.${MASIR_VERSION_MINOR}.${MASIR_VERSION_PATCH}")


###
# Determine where to install python things in our install tree
#
find_package(PythonInterp REQUIRED)
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig; print distutils.sysconfig.get_python_lib(prefix='')"
  RESULT_VARIABLE proc_success
  OUTPUT_VARIABLE python_site_packages
  )

if(NOT ${proc_success} EQUAL 0)
  message(FATAL_ERROR "Request for python site-packages location failed with error code: ${proc_success}")
else()
  string(STRIP "${python_site_packages}" python_site_packages)
  message(STATUS "Python site-packages to install into: ${python_site_packages}")
endif()


###
# Top-level file configuration and installation
#
configure_file("${MASIR_SOURCE_DIR}/setup_masir.build.sh.in"
               "${MASIR_BINARY_DIR}/setup_masir.build.sh"
               @ONLY
               )

configure_file("${MASIR_SOURCE_DIR}/setup_masir.install.sh.in"
               "${MASIR_BINARY_DIR}/setup_masir.install.sh"
               @ONLY
               )

install(
  FILES "${MASIR_SOURCE_DIR}/requirements.txt"
  DESTINATION .
  )
install(
  FILES "${MASIR_BINARY_DIR}/setup_masir.install.sh"
  DESTINATION .
  RENAME setup_masir.sh
  )


add_subdirectory(python)
add_subdirectory(bin)
add_subdirectory(etc)
