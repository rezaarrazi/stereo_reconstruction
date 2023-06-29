#include "stereo_dataset.h"


void StereoDataset::ReadIntrinsics(const std::string& line, std::size_t camera_index)
{

    std::vector<std::size_t> positions;
    std::size_t left_position = 0;
    std::size_t right_position = 0;

    std::vector<double> intrinsics;

    for (std::size_t i = 0; i < line.length(); i++)
    {
        if (line.compare(i, 1, " ") == 0)
            positions.push_back(i);
        else if (line.compare(i, 1, "[") == 0)
            left_position = i;
        else if (line.compare(i, 1, "]") == 0)
            right_position = i;
    }

    for (std::size_t i = 0; i < 9; i++)
    {
        if (i == 0)
            intrinsics.push_back(atof(line.substr(left_position + 1, positions[i] - left_position - 1).c_str()));
        else if (i == 8)
            intrinsics.push_back(atof(line.substr(positions[i - 1] + 1, right_position - positions[i - 1] - 1).c_str()));
        else if (i == 2 || i == 5)
            intrinsics.push_back(atof(line.substr(positions[i - 1] + 1, positions[i] - positions[i - 1] - 2).c_str()));
        else
            intrinsics.push_back(atof(line.substr(positions[i - 1] + 1, positions[i] - positions[i - 1] - 1).c_str()));
    }

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 3; j++)
            camera_intrinsics_[camera_index].at<double>(i, j) = intrinsics[i * 3 + j];
        
}


StereoDataset::StereoDataset()
{

    DIR* dir_ptr = nullptr;
    struct dirent* dirent_ptr = nullptr;

    dir_ptr = opendir(DATA_PATH_.c_str());

    dirent_ptr = readdir(dir_ptr);
    char* folder_name = nullptr;

    while (dirent_ptr != nullptr)
    {
        folder_name = dirent_ptr->d_name;

        if (folder_name[0] == '.')
        {
            dirent_ptr = readdir(dir_ptr);
            continue;
        }

        folder_names_.push_back(std::string(folder_name, folder_name + strlen(folder_name)));

        dirent_ptr = readdir(dir_ptr);
    }

    std::sort(folder_names_.begin(), folder_names_.end());

    image_pair_number_ = folder_names_.size();

    closedir(dir_ptr);
    dir_ptr = nullptr;
    dirent_ptr = nullptr;

}


std::size_t StereoDataset::GetImagePairNumber() const
{
    return image_pair_number_;
}


std::array<cv::Mat, 2> StereoDataset::GetImages() const
{
    return images_;
}


std::array<cv::Mat, 2> StereoDataset::GetDisparityMaps() const
{
    return disparity_maps_;
}


std::array<cv::Mat, 2> StereoDataset::GetCameraIntrinsics() const
{
    return camera_intrinsics_;
}


std::size_t StereoDataset::GetImageWidth() const
{
    return image_width_;
}


std::size_t StereoDataset::GetImageHeight() const
{
    return image_height_;
}


cv::Size StereoDataset::GetImageSize() const
{
    return cv::Size(image_width_, image_height_);
}


std::size_t StereoDataset::GetMinDisparity() const
{
    return min_disparity_;
}


std::size_t StereoDataset::GetMaxDisparity() const
{
    return max_disparity_;
}


void StereoDataset::SetImages(std::size_t image_id)
{

    if (image_id >= image_pair_number_)
        std::cout << "image_id should be < image_pair_number.\n";
    else
    {
        std::string image_pair_path = DATA_PATH_ + "/" + folder_names_[image_id] + "/";
        images_[0] = cv::imread(image_pair_path + "im0.png", CV_LOAD_IMAGE_COLOR);
        images_[1] = cv::imread(image_pair_path + "im1.png", CV_LOAD_IMAGE_COLOR);
    }

}


void StereoDataset::SetDisparityMaps(std::size_t image_id)
{
    if (image_id >= image_pair_number_)
        std::cout << "image_id should be < image_pair_number.\n";
    else
    {
        std::string disparity_map_pair_path = DATA_PATH_ + "/" + folder_names_[image_id] + "/";
        // read PFM file
        disparity_maps_[0] = cv::imread(disparity_map_pair_path + "disp0.pfm", cv::IMREAD_UNCHANGED);
        disparity_maps_[1] = cv::imread(disparity_map_pair_path + "disp1.pfm", cv::IMREAD_UNCHANGED);

        // handle infinite values
        disparity_maps_[0].setTo(0, disparity_maps_[0] == std::numeric_limits<double>::infinity());
        // normalize and convert to uint8
        cv::normalize(disparity_maps_[0], disparity_maps_[0], min_disparity_, max_disparity_, cv::NORM_MINMAX, CV_8U);

        disparity_maps_[1].setTo(0, disparity_maps_[1] == std::numeric_limits<double>::infinity());
        cv::normalize(disparity_maps_[1], disparity_maps_[1], min_disparity_, max_disparity_, cv::NORM_MINMAX, CV_8U);
    }
}


void StereoDataset::SetCalibrations(std::size_t image_id)
{

    if (image_id >= image_pair_number_)
        std::cout << "image_id should be < image_pair_number.\n";
    else
    {
        std::string calibration_path = DATA_PATH_ + "/" + folder_names_[image_id] + "/calib.txt";
        std::ifstream calibration_file(calibration_path);
        std::string line = "";

        getline(calibration_file, line);
        ReadIntrinsics(line, 0);

        getline(calibration_file, line);
        ReadIntrinsics(line, 1);

        getline(calibration_file, line);
        doffs_ = atof(line.substr(line.find("=") + 1).c_str());

        getline(calibration_file, line);
        baseline_ = atof(line.substr(line.find("=") + 1).c_str());

        getline(calibration_file, line);
        image_width_ = static_cast<std::size_t>(std::stoi(line.substr(line.find("=") + 1)));

        getline(calibration_file, line);
        image_height_ = static_cast<std::size_t>(std::stoi(line.substr(line.find("=") + 1)));

        getline(calibration_file, line);
        disparity_num_ = static_cast<std::size_t>(std::stoi(line.substr(line.find("=") + 1)));

        getline(calibration_file, line);
        min_disparity_ = static_cast<std::size_t>(std::stoi(line.substr(line.find("=") + 1)));

        getline(calibration_file, line);
        max_disparity_ = static_cast<std::size_t>(std::stoi(line.substr(line.find("=") + 1)));

        calibration_file.close();
    }

}






