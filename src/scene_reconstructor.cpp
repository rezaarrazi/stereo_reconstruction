#include "scene_reconstructor.h"


cv::Point3f SceneReconstructor::RotateY(const cv::Point3f& point, float theta)
{

    float theta_rad = theta * CV_PI / 180.0;

    return cv::Point3f(cos(theta_rad) * point.x + sin(theta_rad) * point.z,
                       point.y,
                       -sin(theta_rad) * point.x + cos(theta_rad) * point.z);

}


void SceneReconstructor::LoadData(const StereoDataset& stereo_dataset)
{

    double baseline = stereo_dataset.GetBaseline();
    double doffs = stereo_dataset.GetDoffs();
    
    // Get camera matrices
    std::array<cv::Mat, 2> camera_intrinsics = stereo_dataset.GetCameraIntrinsics();
    // Assuming that the camera matrix has type CV_64F (double)
    double f = camera_intrinsics[0].at<double>(0, 0); 

    // initialize Q
    projection_matrix_ = (cv::Mat_<double>(4,4) << 1, 0, 0, -0.5 * stereo_dataset.GetImageWidth(),
                                                   0, -1, 0, 0.5 * stereo_dataset.GetImageHeight(),
                                                   0, 0, 0, -f,
                                                   0, 0, -1 / baseline, doffs / baseline);

}


void SceneReconstructor::ReconstructScene(const cv::Mat& disparity_map, const StereoDataset& stereo_dataset, float distance_threshold)
{

    cv::reprojectImageTo3D(disparity_map, point_cloud_, projection_matrix_);

    float norm = 0.0;
    
    for (std::size_t i = 0; i < point_cloud_.rows; ++i)
    {
        for (std::size_t j = 0; j < point_cloud_.cols; ++j)
        {
            cv::Point3f& point = point_cloud_.at<cv::Point3f>(i, j);

            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                point = RotateY(point, 180);

            norm = cv::norm(point); // calculate the Euclidean norm (magnitude)

            if (norm > distance_threshold)
            {
                point.x = std::numeric_limits<float>::infinity();
                point.y = std::numeric_limits<float>::infinity();
                point.z = std::numeric_limits<float>::infinity();
            }
        }
    }

    // Add color information
    std::array<cv::Mat, 2> images = stereo_dataset.GetImages();
    cv::cvtColor(images[0], images[0], cv::COLOR_BGR2RGB); // Ensure color image is in RGB
    color_cloud_ = images[0].clone();

}


void SceneReconstructor::WriteMeshToFile(const std::string& filename)
{

    float edge_threshold = 50.0f; // 1cm

    std::ofstream out_file(filename);

    if (!out_file)
    {
        std::cerr << "could not open the file!" << std::endl;
        return;
    }

    std::vector<std::array<std::size_t, 3>> triangles; // Triangle indices

    // Prepare data to write
    std::vector<std::pair<cv::Point3f, cv::Vec3b>> valid_points;

    bool valid[4] = {false, false, false, false};

    float d0 = 0.0;
    float d1 = 0.0;
    float d2 = 0.0;

    std::size_t rows = static_cast<std::size_t>(point_cloud_.rows);
    std::size_t cols = static_cast<std::size_t>(point_cloud_.cols);

    for (std::size_t i = 0; i < rows - 2; ++i)
    {
        for (std::size_t j = 0; j < cols - 2; ++j)
        {
            cv::Point3f points[4] = {point_cloud_.at<cv::Point3f>(i, j), point_cloud_.at<cv::Point3f>(i + 1, j),
                                     point_cloud_.at<cv::Point3f>(i, j + 1), point_cloud_.at<cv::Point3f>(i + 1, j + 1)};

            for (std::size_t k = 0; k < 4; ++k)
                valid[k] = std::isfinite(points[k].x) && std::isfinite(points[k].y) && std::isfinite(points[k].z);

            if (valid[0] && valid[1] && valid[2])
            {
                d0 = cv::norm(points[0] - points[1]);
                d1 = cv::norm(points[0] - points[2]);
                d2 = cv::norm(points[1] - points[2]);

                if (edge_threshold > d0 && edge_threshold > d1 && edge_threshold > d2)
                    // triangles.push_back({i * pointCloud.cols + j, (i + 1) * pointCloud.cols + j, i * pointCloud.cols + j + 1});
                    triangles.push_back({i * cols + j, i * cols + j + 1, (i + 1) * cols + j});
            }

            if (valid[1] && valid[2] && valid[3])
            {
                d0 = cv::norm(points[3] - points[1]);
                d1 = cv::norm(points[3] - points[2]);
                d2 = cv::norm(points[1] - points[2]);

                if (edge_threshold > d0 && edge_threshold > d1 && edge_threshold > d2)
                    // triangles.push_back({(i + 1) * pointCloud.cols + j, (i + 1) * pointCloud.cols + j + 1, i * pointCloud.cols + j + 1});
                    triangles.push_back({(i + 1) * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1});
            }
        }
    }

    // Write the .off header
    out_file << "COFF\n" << rows * cols << " " << triangles.size() << " " << 0 << "\n";
    
    // Write vertices
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            cv::Point3f point = point_cloud_.at<cv::Point3f>(i, j);
            cv::Vec3b color = color_cloud_.at<cv::Vec3b>(i, j);

            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                out_file << point.x << " " << point.y << " " << point.z << " " << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << " 255\n";
            else
                out_file << "0.0 0.0 0.0 0 0 0 0\n";
        }
    }

    out_file << "# list of faces" << std::endl;
	out_file << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    // Write triangles
    for (const auto& triangle: triangles)
        out_file << "3 " << triangle[0] << " " << triangle[1] << " " << triangle[2] << "\n";

    out_file.close();

}




