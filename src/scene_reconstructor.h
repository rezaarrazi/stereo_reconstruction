#ifndef _SCENE_RECONSTRUCTOR
#define _SCENE_RECONSTRUCTOR


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "stereo_dataset.h"


class SceneReconstructor
{

    private:
        cv::Mat projection_matrix_; // projection matrix
        cv::Mat point_cloud_; // 3D point cloud representation
        cv::Mat color_cloud_; // Color information for each point in the point cloud

        cv::Point3f RotateY(const cv::Point3f& point, float theta);

    public:
        SceneReconstructor() {}

        void LoadData(const StereoDataset& stereo_dataset);

        void ReconstructScene(const cv::Mat& disparity_map, const StereoDataset& stereo_dataset, float distance_threshold);

        void WriteMeshToFile(const std::string& filename);

};


#endif




