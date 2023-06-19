#ifndef _STEREO_DATASET
#define _STEREO_DATASET


#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>


class StereoDataset
{

    private:
        const std::string DATA_PATH = "../Data/Middlebury";
        std::vector<std::string> folder_names_;
        std::size_t image_pair_num_ = 0;

    public:
        StereoDataset();

        std::array<std::string, 2> GetImagePair(int image_id) const;

};


#endif








