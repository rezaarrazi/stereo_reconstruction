#include "scene_reconstructor.h"


cv::Point3f SceneReconstructor::RotateY(const cv::Point3f& point, float theta)
{

    float theta_rad = theta * CV_PI / 180.0;

    return cv::Point3f(cos(theta_rad) * point.x + sin(theta_rad) * point.z,
                       point.y,
                       -sin(theta_rad) * point.x + cos(theta_rad) * point.z);

}


bool SceneReconstructor::AreDistancesValid(Vertex* vertices, std::size_t index0, std::size_t index1, std::size_t index2, float distance_threshold)
{
    bool flag0 = (vertices[index0].position_ - vertices[index1].position_).norm() < distance_threshold;
    bool flag1 = (vertices[index1].position_ - vertices[index2].position_).norm() < distance_threshold;
    bool flag2 = (vertices[index2].position_ - vertices[index0].position_).norm() < distance_threshold;
    return flag0 && flag1 && flag2;
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

    images_[0] = stereo_dataset.GetImages()[0].clone();
    images_[1] = stereo_dataset.GetImages()[1].clone();

}


void SceneReconstructor::ReconstructScene(const cv::Mat& disparity_map, float distance_threshold)
{

    cv::reprojectImageTo3D(disparity_map, point_cloud_, projection_matrix_);

    float norm = 0.0;
    
    for (std::size_t i = 0; i < point_cloud_.rows; ++i)
    {
        for (std::size_t j = 0; j < point_cloud_.cols; ++j)
        {
            cv::Point3f& point = point_cloud_.at<cv::Point3f>(i, j);

            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                point.z *= -1.0f;
                // point = RotateY(point, 180);

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
    cv::Mat color_tmp = images_[0].clone();
    cv::cvtColor(color_tmp, color_tmp, cv::COLOR_BGR2RGB); // Ensure color image is in RGB
    color_cloud_ = color_tmp;

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
                    triangles.push_back({i * point_cloud_.cols + j, (i + 1) * point_cloud_.cols + j, i * point_cloud_.cols + j + 1});
                    // triangles.push_back({i * cols + j, i * cols + j + 1, (i + 1) * cols + j});
            }

            if (valid[1] && valid[2] && valid[3])
            {
                d0 = cv::norm(points[3] - points[1]);
                d1 = cv::norm(points[3] - points[2]);
                d2 = cv::norm(points[1] - points[2]);

                if (edge_threshold > d0 && edge_threshold > d1 && edge_threshold > d2)
                    triangles.push_back({(i + 1) * point_cloud_.cols + j, (i + 1) * point_cloud_.cols + j + 1, i * point_cloud_.cols + j + 1});
                    // triangles.push_back({(i + 1) * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1});
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


void SceneReconstructor::ReconstructSceneDirectly(const cv::Mat& disparity_map, const StereoDataset& stereo_dataset,
                                                  float distance_threshold, const std::string& file_name)
{

    cv::Mat image0 = stereo_dataset.GetImages()[0];

    cv::Mat camera_intrinsics = stereo_dataset.GetCameraIntrinsics()[0];

    float focal = static_cast<float>(camera_intrinsics.at<double>(0, 0));
    float center_x = static_cast<float>(camera_intrinsics.at<double>(0, 2));
    float center_y = static_cast<float>(camera_intrinsics.at<double>(1, 2));
    float baseline = static_cast<float>(stereo_dataset.GetBaseline());

    // cv::Mat depth_map = 16 * focal * baseline / disparity_map;

    cv::Mat depth_map = cv::Mat::zeros(stereo_dataset.GetImageSize(), CV_16SC1);

    std::size_t rows = depth_map.rows;
    std::size_t cols = depth_map.cols;

    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < cols; j++)
        {
            if (disparity_map.at<short>(i, j) == 0)
                depth_map.at<short>(i, j) = 0;
            else
                depth_map.at<short>(i, j) = 16 * focal * baseline / disparity_map.at<short>(i, j);
        }
    }

    for (std::size_t i = 0; i < rows; i++)
        for (std::size_t j = 0; j < cols; j++)
            if (depth_map.at<short>(i, j) == -32768)
                depth_map.at<short>(i, j) = 32767;

    Vertex* vertices = new Vertex[rows * cols];

    float depth = 0.0;

    float x = 0.0;
    float y = 0.0;

    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < cols; j++)
        {
            depth = static_cast<float>(depth_map.at<short>(i, j));

            if (depth != MINF && depth != 0.0 && depth < 32768.0)
            {
                x = (j - center_x) * depth / focal;
                y = (i - center_y) * depth / focal;

                vertices[i * cols + j].position_ = Vector4f(x, y, depth, 1.0);

                vertices[i * cols + j].color_ = Vector4uc(image0.at<cv::Vec3b>(i, j)[2],
                                                          image0.at<cv::Vec3b>(i, j)[1],
                                                          image0.at<cv::Vec3b>(i, j)[0],
                                                          255);
            }
            else
            {
                vertices[i * cols + j].position_ = Vector4f(MINF, MINF, MINF, MINF);
                vertices[i * cols + j].color_ = Vector4uc(0, 0, 0, 0);
            }
        }
    }

    std::vector<std::array<std::size_t, 3>> faces;

    std::size_t face_number = 0;

    std::array<std::size_t, 4> indices = {0, 0, 0, 0};

    std::array<bool, 4> valid_conditions = {false, false, false, false};

    for (std::size_t i = 0; i < rows - 1; i++)
    {
        for (std::size_t j = 0; j < cols - 1; j++)
        {
            indices[0] = i * cols + j;
            indices[1] = (i + 1) * cols + j;
            indices[2] = i * cols + j + 1;
            indices[3] = (i + 1) * cols + j + 1;

            for (std::size_t k = 0; k < 4; k++)
                valid_conditions[k] = vertices[indices[k]].position_[0] != MINF;
            
            if (valid_conditions[0] == true && valid_conditions[1] == true && valid_conditions[2] == true)
            {
                if (AreDistancesValid(vertices, indices[0], indices[1], indices[2], distance_threshold) == true)
                {
                    faces.push_back(std::array<std::size_t, 3>{indices[0], indices[1], indices[2]});
                    face_number++;
                }
            }

            if (valid_conditions[1] == true && valid_conditions[3] == true && valid_conditions[2] == true)
            {
                if (AreDistancesValid(vertices, indices[1], indices[3], indices[2], distance_threshold) == true)
                {
                    faces.push_back(std::array<std::size_t, 3>{indices[1], indices[3], indices[2]});
                    face_number++;
                }
            }
        }
    }

    std::ofstream out_file(file_name);

    out_file << "COFF" << std::endl;
    out_file << rows * cols << " " << face_number << " 0" << std::endl;

    for (std::size_t i = 0; i < rows * cols; i++)
    {
        if (vertices[i].position_[0] == MINF)
            out_file << "0.0 0.0 0.0 ";
        else
            for (std::size_t j = 0; j < 3; j++)
                out_file << vertices[i].position_[j] << " ";

        for (std::size_t j = 0; j < 3; j++)
            out_file << static_cast<int>(vertices[i].color_[j]) << " ";

        out_file << static_cast<int>(vertices[i].color_[3]) << std::endl;
    }

    for (std::size_t i = 0; i < faces.size(); i++)
        out_file << "3 " << faces[i][0] << " " << faces[i][1] << " " << faces[i][2] << " " << std::endl;
    
    out_file.close();

    delete[] vertices;
    vertices = nullptr;

}




