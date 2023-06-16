# Stereo Vision Project

## Overview
This project is a stereo vision application that loads and processes images from a specific dataset. The application is developed in C++ and uses the Eigen, Google Log (glog), Ceres Solver, and OpenCV libraries.

The application loads stereo pairs, ground truth disparity maps (.pfm files), and calibration data from the [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2021/). The images are then processed to compute disparity maps using a stereo vision algorithm.

## Prerequisites

* A modern C++ compiler (C++14 or higher is required)
* CMake (version 3.13 or higher is required)
* Eigen library
* Google Log (glog) library
* Ceres Solver library
* OpenCV library

## Building the Project

1. Clone the repository to your local machine.
2. Inside the project directory, create a new directory called `build`.
3. Inside the `build` directory, run `cmake ..` to generate the Makefile.
4. Run `make` to compile the project.

## Running the Application

After building the project, you can run the application with the command `./stereo`.

## Dataset

The dataset used for this project can be downloaded from the following link: [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2021/)
