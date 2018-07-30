![las2peer](https://rwth-acis.github.io/las2peer/logo/vector/las2peer-logo.svg)

# las2peer-TensorFlow-Classifier

[![Build Status](https://travis-ci.org/rwth-acis/las2peer-TensorFlow-Classifier-Service.svg?branch=master)](https://travis-ci.org/rwth-acis/las2peer-TensorFlow-Classifier-Service) [![codecov](https://codecov.io/gh/rwth-acis/las2peer-TensorFlow-Classifier-Service/branch/master/graph/badge.svg)](https://codecov.io/gh/rwth-acis/las2peer-TensorFlow-Classifier-Service) 

This las2peer service is a wrapper for a customized version of the [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf). It is an implementation of a convolutional neural network for text classification. So far only the inference method has been implemented in Java, because with the current version of TensorFlow (r1.9) the training of deep learning models in Java is not yet possible. 
The service implements the [BotContentGenerator interface](https://github.com/rwth-acis/las2peer/blob/master/core/src/main/java/i5/las2peer/logging/bot/BotContentGenerator.java). 

Build
--------
Execute the following command on your shell:

```shell
ant all 
```

Start
--------

To start the data-processing service, use one of the available start scripts:

Windows:

```shell
bin/start_network.bat
```

Unix/Mac:
```shell
bin/start_network.sh
```
