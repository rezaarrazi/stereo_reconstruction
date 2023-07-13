#include <iostream>
#include <fstream>
#include "mesh_reconstruction.h"

MeshReconstruction::MeshReconstruction(StereoDataset dataset) {
    double baseline = dataset.GetBaseLine();
    double doffs = dataset.GetDoffs();
    
    // Get camera matrices
    std::array<cv::Mat, 2> camMats = dataset.GetCameraIntrinsics();
    // Assuming that the camera matrix has type CV_64F (double)
    double f = camMats[0].at<double>(0, 0); 

    // initialize Q
    Q = (cv::Mat_<double>(4,4) << 
        1, 0, 0, -0.5 * dataset.GetImageWidth(), 
        0, -1, 0, 0.5 * dataset.GetImageHeight(), 
        0, 0, 0, -f, 
        0, 0, -1/baseline, doffs / baseline);
}

void MeshReconstruction::reconstructMesh(const cv::Mat& disparityMap, StereoDataset dataset) {
    cv::reprojectImageTo3D(disparityMap, pointCloud, Q);
    
    // Filter out invalid points
    for (int i = 0; i < pointCloud.rows; ++i) {
        for (int j = 0; j < pointCloud.cols; ++j) {
            cv::Point3f& point = pointCloud.at<cv::Point3f>(i, j);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                point.x = 0;
                point.y = 0;
                point.z = 0;
            }
        }
    }
    
    // Add color information
    std::array<cv::Mat, 2> images = dataset.GetImages();
    cv::cvtColor(images[0], images[0], cv::COLOR_BGR2RGB); // Ensure color image is in RGB
    colorCloud = images[0].clone();
}

void MeshReconstruction::writeMeshToFile(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    // Prepare data to write
    std::vector<std::pair<cv::Point3f, cv::Vec3b>> validPoints;
    for (int i = 0; i < pointCloud.rows; ++i) {
        for (int j = 0; j < pointCloud.cols; ++j) {
            cv::Point3f point = pointCloud.at<cv::Point3f>(i, j);
            cv::Vec3b color = colorCloud.at<cv::Vec3b>(i, j);
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                validPoints.push_back({point, color});
            }
        }
    }

    // Write the .off header
    outFile << "COFF\n" << validPoints.size() << " " << 0 << " " << 0 << "\n";
    
    // Write valid points
    for (const auto& [point, color] : validPoints) {
        outFile << point.x << " " << point.y << " " << point.z << " " << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << "\n";
    }

    outFile.close();
}

