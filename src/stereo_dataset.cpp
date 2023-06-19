#include "stereo_dataset.h"


StereoDataset::StereoDataset()
{

    DIR* dir_ptr = nullptr;
    struct dirent* dirent_ptr = nullptr;

    dir_ptr = opendir(DATA_PATH.c_str());

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

    image_pair_num_ = folder_names_.size();

    closedir(dir_ptr);
    dir_ptr = nullptr;
    dirent_ptr = nullptr;

}


std::array<std::string, 2> StereoDataset::GetImagePair(int image_id) const
{
    std::array<std::string, 2> image_pairs;

    if (image_id < 0 || image_id >= image_pair_num_)
    {
        std::cout << "image_id should be >= 0 and < image_pair_num.\n";
        return image_pairs;
    }

    std::string image_pair_path = DATA_PATH + "/" + folder_names_[image_id] + "/";

    image_pairs[0] = image_pair_path + "im0.png";
    image_pairs[1] = image_pair_path + "im1.png";

    return image_pairs;
}






