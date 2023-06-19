#ifndef _SPARSE_MATCHER
#define _SPARSE_MATCHER


#include <vector>
#include <array>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>


class SparseMatcher
{

    private:
        std::size_t match_num_ = 300;
        cv::Ptr<cv::DescriptorMatcher> feature_matcher_;
        std::vector<cv::DMatch> matches_;
        std::vector<cv::Point2f> matched_points0_;
        std::vector<cv::Point2f> matched_points1_;
        std::vector<cv::DMatch> selected_matches_;
        static bool CompareMatches(const cv::DMatch& match0, const cv::DMatch& match1);

    public:
        SparseMatcher();

        void SetMatchNum(std::size_t match_num);

        void MatchSparsely(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features);

        void DisplayMatchings(const cv::Mat& image0, const cv::Mat& image1, const std::array<std::vector<cv::KeyPoint>, 2>& keypoints);

};


#endif






