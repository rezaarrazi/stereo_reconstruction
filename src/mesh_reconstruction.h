#ifndef _MESH_RECONSTRUCTION
#define _MESH_RECONSTRUCTION

#include <opencv2/opencv.hpp>
#include "stereo_dataset.h"

class MeshReconstruction {
private:
    cv::Mat Q; // projection matrix
    cv::Mat pointCloud; // 3D point cloud representation
    cv::Mat colorCloud; // Color information for each point in the point cloud

    cv::Point3f rotateY(const cv::Point3f& point, float theta);
public:
    MeshReconstruction(StereoDataset dataset);

    void reconstructMesh(const cv::Mat& disparityMap, StereoDataset dataset);
    void writeMeshToFile(const std::string& filename);
};

#endif