# Install scripts

install(
  PROGRAMS classifier_kfold_validation.py
           classifier_model_validation.py
           classifyFiles.py
           compute_classifications.py
           compute_hash_codes.py
           compute_many_descriptors.py
           computeDescriptor.py
           createFileIngest.py
           descriptors_to_svmtrainfile.py
           generate_image_transform.py
           iqr_app_model_generation.py
           iqrTrainClassifier.py
           proxyManagerServer.py
           removeOldFiles.py
           runApplication.py
           summarizePlugins.py
           train_itq.py
  DESTINATION bin
  )
