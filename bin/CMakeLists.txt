# Install pre-compiled binaries

install(
  PROGRAMS classifier_kfold_validation.py
           classifyFiles.py
           computeDescriptor.py
           createFileIngest.py
           iqr_app_model_generation.py
           iqrTrainClassifier.py
           proxyManagerServer.py
           removeOldFiles.py
           runApplication.py
           summarizePlugins.py
  DESTINATION bin
  )
