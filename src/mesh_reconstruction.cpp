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
    
    std::vector<std::pair<cv::Point3f, cv::Vec3b>> pointVertices;
    for (int i = 0; i < pointCloud.rows; ++i) {
        for (int j = 0; j < pointCloud.cols; ++j) {
            cv::Point3f point = pointCloud.at<cv::Point3f>(i, j);
            cv::Vec3b color = colorCloud.at<cv::Vec3b>(i, j);

            pointVertices.push_back({point, color});
        }
    }

    std::vector<std::array<int, 3>> triangles; // Triangle indices

    // Prepare data to write
    int height = pointCloud.rows;
    int width = pointCloud.cols;
    for (int i = 0; i < height-1; ++i) {
        for (int j = 0; j < width-1; ++j) {
            int i0 = i*width + j;
			int i1 = (i + 1)*width + j;
			int i2 = i*width + j + 1;
			int i3 = (i + 1)*width + j + 1;

            cv::Point3f points[4] = {
                pointVertices[i0].first,
                pointVertices[i1].first,
                pointVertices[i2].first,
                pointVertices[i3].first
            };

            bool valid[4];
            for (int k = 0; k < 4; ++k)
                valid[k] = std::isfinite(points[k].x) && std::isfinite(points[k].y) && std::isfinite(points[k].z);

            if (valid[0] && valid[1] && valid[2]) {
                float d0 = cv::norm(points[0] - points[1]);
                float d1 = cv::norm(points[0] - points[2]);
                float d2 = cv::norm(points[1] - points[2]);

                // std::cout << "d0: " << d0 << ", d1: " << d1 << ", d2: " << d2 << std::endl; 

                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2){
                    // std::cout << "p0: " << pointVertices[i0].first.x << ", " << pointVertices[i0].first.y << ", p1: " << pointVertices[i1].first.x << ", " << pointVertices[i1].first.y << ", p2: " << pointVertices[i2].first.x << ", " << pointVertices[i2].first.y << std::endl; 
                    triangles.push_back({i0, i1, i2});
                }
                
            }

            if (valid[1] && valid[2] && valid[3]) {
                float d0 = cv::norm(points[3] - points[1]);
                float d1 = cv::norm(points[3] - points[2]);
                float d2 = cv::norm(points[1] - points[2]);

                // std::cout << "d0: " << d0 << ", d1: " << d1 << ", d2: " << d2 << std::endl; 

                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2){
                    // std::cout << "p0: " << pointVertices[i0].first.x << ", " << pointVertices[i0].first.y << ", p1: " << pointVertices[i1].first.x << ", " << pointVertices[i1].first.y << ", p2: " << pointVertices[i2].first.x << ", " << pointVertices[i2].first.y << std::endl; 
                    triangles.push_back({i1, i3, i2});
                }
            }
        }
    }

    size_t faces_limit = 100;
    // Write the .off header
    outFile << "COFF\n" << pointCloud.rows*pointCloud.cols << " " << triangles.size() << " " << 0 << "\n";
    
    // Write vertices
    for (const auto& [point, color] : pointVertices) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            outFile << point.x << " " << point.y << " " << point.z << " " 
                    << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << " 255\n";
        }
        else {
            outFile << "0.0 0.0 0.0 0 0 0 0\n";
        }
    }

    outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    // Write triangles
    for (size_t i = 0; i < triangles.size(); ++i) {
        // unsigned int i0 = triangles[i][0];
        // unsigned int i1 = triangles[i][1];
        // unsigned int i2 = triangles[i][2];
        
        // cv::Point3f p0 = pointVertices[i0].first;
        // cv::Point3f p1 = pointVertices[i1].first;
        // cv::Point3f p2 = pointVertices[i2].first;

        // float d0 = cv::norm(p0 - p1);
        // float d1 = cv::norm(p0 - p2);
        // float d2 = cv::norm(p1 - p2);

        // std::cout << "i0: " << i0 / width << ", " << i0 % width << ", i1: " << i1 / width << ", " << i1 % width << ", i2: " << i2 / width << ", " << i2 % width << std::endl; 
        // std::cout << "p0: " << p0.x << ", " << p0.y << ", p1: " << p1.x << ", " << p1.y << ", p2: " << p2.x << ", " << p2.y << std::endl; 
        // std::cout << "d0: " << d0 << ", d1: " << d1 << ", d2: " << d2 << std::endl; 
        // std::cout << "------------------------" << std::endl; 

        outFile << "3 " << triangles[i][0] << " " << triangles[i][1] << " " << triangles[i][2] << "\n";
    }

    outFile.close();
}

