# Stereo Reconstruction Project

## Overview
This project is a stereo reconstruction project that aims at reconstructing 3D scenes from a pair of images and comparing the performance of 
different algorithms. The project is developed in C++ and uses OpenCV and libtorch. 

## Dataset 

The dataset used in the project is the [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2021/). 

## Prerequisites

* A modern C++ compiler (C++11 or higher is required)
* CMake (version 2.8 or higher is required)
* OpenCV
* [libtorch](https://pytorch.org/cppdocs/installing.html)

## Building the Project

1. Clone the repository to your local machine.
2. [*Installing C++ Distributions of PyTorch*](https://pytorch.org/cppdocs/installing.html) for `libtorch` setup.
2. Inside the project directory, create a new directory called `build`.
3. Inside the `build` directory, run `cmake .. -DCMAKE_PREFIX_PATH=<libtorch path>` to generate the Makefile.
4. Run `make` to compile the project.

## Running the Application

After building the project, you can run the executable files in the `build` folder. 

## The file structures

There are four main folders in this repository. 

The folder **Data** is used to store the data. We do not upload the data due to the file size, the data can be found using the link provided above. 

The folder **figures** contains the figures needed for the README.md. 

The folder **results** contains some results produced by the pipeline. 

The folder **src** contains the source code. 

## Source code

The two main function files are **experiment.cpp** and **test.cpp**. These two files are mainly used for experimens and testing. 

The folder **superglue** contains files used for **SuperGlue**. 

**PFMReadWrite.h** and **PFMReadWrite.cpp**: These files are used to read the ground truth disparity maps provided by the dataset. 

**stereo_dataset.h** and **stereo_dataset.cpp**: Header file and C++ implementation file for the class **StereoDataset**. The class is used to read the data contained in the Data folder and output them. 

**feature_extractor.h** and **feature_extractor.cpp**: Header file and C++ implementation file for the class **FeatureExtractor**. The class is used for feature extraction part. 

**sparse_matcher.h** and **sparse_matcher.cpp**: Header file and C++ implementation file for the class **SparseMatcher**. The class is used for sparse matching part. 

**camera_pose_estimator.h** and **camera_pose_estimator.cpp**: Header file and C++ implementation file for the class **CameraPoseEstimator**. The class is used to estimate the camera poses. 

**dense_matcher.h** and **dense_matcher.cpp**: Header file and C++ implementation file for the class **DenseMatcher**. The class is used for dense matching part. 

**scene_reconstructor.h** and **scene_reconstructor.cpp**: Header file and C++ implementation file for the class **SceneReconstructor**. The class is used to reconstruct the scenes. 

**experiment_designer.h** and **experiment_designer.cpp**: Header file and C++ implementation file for the class **ExperimentDesigner**. The class is used to design experiments. 






