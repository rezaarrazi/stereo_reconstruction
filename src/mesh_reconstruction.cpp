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

cv::Point3f MeshReconstruction::rotateY(const cv::Point3f& point, float theta) {
    float thetaRad = theta * CV_PI / 180.0;
    return cv::Point3f(
        cos(thetaRad) * point.x + sin(thetaRad) * point.z,
        point.y,
        -sin(thetaRad) * point.x + cos(thetaRad) * point.z
    );
}

void MeshReconstruction::reconstructMesh(const cv::Mat& disparityMap, StereoDataset dataset) {
    cv::reprojectImageTo3D(disparityMap, pointCloud, Q);
    
    for (int i = 0; i < pointCloud.rows; ++i) {
        for (int j = 0; j < pointCloud.cols; ++j) {
            cv::Point3f& point = pointCloud.at<cv::Point3f>(i, j);
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                point = rotateY(point, 180);
            }
        }
    }

    // Add color information
    std::array<cv::Mat, 2> images = dataset.GetImages();
    cv::cvtColor(images[0], images[0], cv::COLOR_BGR2RGB); // Ensure color image is in RGB
    colorCloud = images[0].clone();
}

void MeshReconstruction::writeMeshToFile(const std::string& filename) {
    float edgeThreshold = 50.0f; // 1cm

    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    std::vector<std::array<int, 3>> triangles; // Triangle indices

    // Prepare data to write
    std::vector<std::pair<cv::Point3f, cv::Vec3b>> validPoints;
    for (int i = 0; i < pointCloud.rows-2; ++i) {
        for (int j = 0; j < pointCloud.cols-2; ++j) {
            cv::Point3f points[4] = {
                pointCloud.at<cv::Point3f>(i, j),
                pointCloud.at<cv::Point3f>(i+1, j),
                pointCloud.at<cv::Point3f>(i, j+1),
                pointCloud.at<cv::Point3f>(i+1, j+1)
            };

            bool valid[4];
            for (int k = 0; k < 4; ++k)
                valid[k] = std::isfinite(points[k].x) && std::isfinite(points[k].y) && std::isfinite(points[k].z);

            if (valid[0] && valid[1] && valid[2]) {
                float d0 = cv::norm(points[0] - points[1]);
                float d1 = cv::norm(points[0] - points[2]);
                float d2 = cv::norm(points[1] - points[2]);

                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    // triangles.push_back({i * pointCloud.cols + j, (i + 1) * pointCloud.cols + j, i * pointCloud.cols + j + 1});
                    triangles.push_back({i * pointCloud.cols + j, i * pointCloud.cols + j + 1, (i + 1) * pointCloud.cols + j});
            }

            if (valid[1] && valid[2] && valid[3]) {
                float d0 = cv::norm(points[3] - points[1]);
                float d1 = cv::norm(points[3] - points[2]);
                float d2 = cv::norm(points[1] - points[2]);

                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    // triangles.push_back({(i + 1) * pointCloud.cols + j, (i + 1) * pointCloud.cols + j + 1, i * pointCloud.cols + j + 1});
                    triangles.push_back({(i + 1) * pointCloud.cols + j, i * pointCloud.cols + j + 1, (i + 1) * pointCloud.cols + j + 1});
            }
        }
    }

    // Write the .off header
    outFile << "COFF\n" << pointCloud.rows*pointCloud.cols << " " << triangles.size() << " " << 0 << "\n";
    
    // Write vertices
    for (int i = 0; i < pointCloud.rows; ++i) {
        for (int j = 0; j < pointCloud.cols; ++j) {
            cv::Point3f point = pointCloud.at<cv::Point3f>(i, j);
            cv::Vec3b color = colorCloud.at<cv::Vec3b>(i, j);

            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                outFile << point.x << " " << point.y << " " << point.z << " " 
                        << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << " 255\n";
            }
            else {
                outFile << "0.0 0.0 0.0 0 0 0 0\n";
            }
        }
    }

    outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    // Write triangles
    for (const auto& triangle : triangles) {
        outFile << "3 " << triangle[0] << " " << triangle[1] << " " << triangle[2] << "\n";
    }

    outFile.close();
}

