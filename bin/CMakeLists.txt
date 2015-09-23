# Install pre-compiled binaries

install(
  PROGRAMS computeDescriptor.py
           createFileIngest.py
           proxyManagerServer.py
           removeOldFiles.py
           runApplication.py
  DESTINATION bin
  )
