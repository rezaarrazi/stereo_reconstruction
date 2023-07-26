#ifndef _SCENE_RECONSTRUCTOR
#define _SCENE_RECONSTRUCTOR


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "Eigen.h"
#include "stereo_dataset.h"


class Vertex
{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Vector4f position_;
        Vector4uc color_;

};


class SceneReconstructor
{

    private:
        cv::Mat projection_matrix_; // projection matrix
        cv::Mat point_cloud_; // 3D point cloud representation
        cv::Mat color_cloud_; // Color information for each point in the point cloud

        cv::Point3f RotateY(const cv::Point3f& point, float theta);

        bool AreDistancesValid(Vertex* vertices, std::size_t index0, std::size_t index1, std::size_t index2, float distance_threshold);

    public:
        SceneReconstructor() {}

        void LoadData(const StereoDataset& stereo_dataset);

        void ReconstructScene(const cv::Mat& disparity_map, const StereoDataset& stereo_dataset, float distance_threshold);

        void WriteMeshToFile(const std::string& filename);

        void ReconstructSceneDirectly(const cv::Mat& disparity_map, const StereoDataset& stereo_dataset,
                                      float distance_threshold, const std::string& file_name);

};


#endif




