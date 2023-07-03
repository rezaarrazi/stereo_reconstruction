# Stereo Reconstruction Project

## Overview
This project is a stereo reconstruction project that aims at reconstructing 3D scenes from a pair of images and comparing the performance of 
different algorithms. The project is developed in C++ and uses OpenCV, Eigen, Google Log (glog), Ceres Solver, and g2o libraries. 

## Dataset 

The dataset used in the project is the [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2021/). 

## Prerequisites

* A modern C++ compiler (C++11 or higher is required)
* CMake (version 2.8 or higher is required)
* OpenCV
* Eigen
* Google Log (glog)
* Ceres Solver
* g2o 
* [libtorch] (https://pytorch.org/cppdocs/installing.html)

## Building the Project

1. Clone the repository to your local machine.
2. [*Installing C++ Distributions of PyTorch*](https://pytorch.org/cppdocs/installing.html) for `libtorch` setup.
2. Inside the project directory, create a new directory called `build`.
3. Inside the `build` directory, run `cmake .. -DCMAKE_PREFIX_PATH=<libtorch path>` to generate the Makefile.
4. Run `make` to compile the project.

## Running the Application

After building the project, you can run the executable files in the `build` folder. 


