project(SMQTK)
cmake_minimum_required(VERSION 2.8)


###
# Versioning
#
set(SMQTK_VERSION_MAJOR 1)
set(SMQTK_VERSION_MINOR 1)
set(SMQTK_VERSION_PATCH 0)
set(SMQTK_VERSION ${SMQTK_VERSION_MAJOR}.${SMQTK_VERSION_MINOR}.${SMQTK_VERSION_PATCH})


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
# TPL
#
# Location variables
set(TPL_BUILD_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/TPL/build")
set(TPL_LOCAL_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/TPL/install")
set(TPL_PYTHON_SP "${TPL_LOCAL_INSTALL}/${python_site_packages}")
add_subdirectory( TPL )


###
# SMQTK Library
#
add_subdirectory( lib )
add_subdirectory( bin )


###
# MASIR side-car project
#
add_subdirectory(MASIR)

###
# File configuration
#
# Environment setup scripts
configure_file("${SMQTK_SOURCE_DIR}/setup_env.build.sh.in"
               "${SMQTK_BINARY_DIR}/setup_env.build.sh"
               @ONLY
               )
configure_file("${SMQTK_SOURCE_DIR}/setup_env.install.sh.in"
               "${SMQTK_BINARY_DIR}/setup_env.install.sh"
               @ONLY
               )

# Default system configuration file
configure_file("${SMQTK_SOURCE_DIR}/smqtk_config.build.py.in"
               "${SMQTK_SOURCE_DIR}/smqtk_config.py"
               @ONLY
               )
configure_file("${SMQTK_SOURCE_DIR}/smqtk_config.install.py.in"
               "${SMQTK_BINARY_DIR}/smqtk_config.install.py"
               @ONLY
               )


###
# System Installation
#

option(FORCE_STANDARD_INSTALL_PREFIX
  "Force the standard install prefix (for initial override of basic install prefix)."
  ON
  )
if (FORCE_STANDARD_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/opt/kitware/SMQTK"
    CACHE PATH
    "Install path prefix, prepended onto install directories."
    FORCE)
endif()

#+
# inital empty data and work directories
install(DIRECTORY DESTINATION var/work)
install(DIRECTORY DESTINATION var/data)
install(DIRECTORY DESTINATION var/nltk_data)
install(DIRECTORY static/ DESTINATION var/static)

if(UNIX)
  # Due to super-user privaleges being required to run things like groupadd,
  # these commands will only be executed when we have such permissions (like
  # installing to the system, etc.)
  install(CODE
          "execute_process(COMMAND groupadd -r SMQTK
                           COMMAND chown -R :SMQTK var
                           COMMAND chmod -R 2775 var
                           WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\")"
          )
endif()
#-

install(DIRECTORY   docs/
        DESTINATION share/doc/SMQTK
        )

install(DIRECTORY   etc/
        DESTINATION etc
        )

install(FILES requirements.txt
        DESTINATION .
        )

install(FILES "${SMQTK_BINARY_DIR}/setup_env.install.sh"
        DESTINATION .
        RENAME setup_smqtk.sh
        )

install(FILES "${SMQTK_BINARY_DIR}/smqtk_config.install.py"
        DESTINATION "${python_site_packages}"
        RENAME smqtk_config.py
        )

install(DIRECTORY WebUI
        DESTINATION "${python_site_packages}"
        )

install(DIRECTORY common_utils
        DESTINATION "${python_site_packages}"
        )

install(PROGRAMS run_WebApp.py
        DESTINATION bin
        )

install(PROGRAMS run_static_server.py
        DESTINATION bin
        )

install(PROGRAMS run_smqtk.py
        DESTINATION bin
        )

###
# CPack stuff
#

# TODO: Define package dependencies here
set(SMQTK_DEPS "")

if(EXISTS /etc/redhat-release)
  file(READ /etc/redhat-release RHEL_VERSION)
  string(REGEX REPLACE ".*release ([^\\. ]*).*" "\\1" RHEL_VERSION "${RHEL_VERSION}")
  set(CPACK_SYSTEM_NAME "el${RHEL_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_RPM_PACKAGE_AUTOREQPROV "no")
  set(CPACK_RPM_PACKAGE_REQUIRES "${SMQTK_DEPS}")
  set(CPACK_RPM_PACKAGE_RELOCATABLE TRUE)
else()
  set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CPACK_PACKAGE_NAME "SMQTK")
set(CPACK_PACKAGE_VENDOR "Kitware, Inc.")
set(CPACK_PACKAGE_CONTACT "kitware@kitware.com")
#set(CPACK_MONOLITHIC_INSTALL true)
set(CPACK_PACKAGE_VERSION_MAJOR ${SMQTK_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SMQTK_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SMQTK_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION       ${SMQTK_VERSION})
#set(CPACK_RESOURCE_FILE_LICENSE [NONE YET])
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")

include(CPack)
