#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>


class StereoDataset
{
    private:
        const std::string DATA_PATH = "../Data/Middlebury";
        std::vector<std::string> folder_names;
        std::size_t image_pair_num = 0;
    public:
        StereoDataset();
        void TestDataset();
        std::vector<std::string> GetImagePair(int image_id) const;
};








