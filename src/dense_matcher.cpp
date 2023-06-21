#include "dense_matcher.h"


cv::Mat DenseMatcher::GetDisparityMap() const
{
    return disparity_map_;
}


void DenseMatcher::ComputeDisparityMap(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation)
{
    std::array<cv::Mat, 2> camera_intrinsics = stereo_dataset.GetCameraIntrinsics();

    // rectify images
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::stereoRectify(camera_intrinsics[0], cv::noArray(), camera_intrinsics[1], cv::noArray(), stereo_dataset.GetImageSize(),
                      rotation, translation, R1_, R2_, P1_, P2_, Q_);
    
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(camera_intrinsics[1], cv::noArray(), R1_, P1_, stereo_dataset.GetImageSize(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(camera_intrinsics[0], cv::noArray(), R2_, P2_, stereo_dataset.GetImageSize(), CV_16SC2, map21, map22);

    cv::remap(stereo_dataset.GetImages()[0], rectified_images_[0], map11, map12, cv::INTER_CUBIC);
    cv::remap(stereo_dataset.GetImages()[1], rectified_images_[1], map21, map22, cv::INTER_CUBIC);
    
    // create a StereoSGBM object
    int window_size = 5;
    int disparity_num = (stereo_dataset.GetMaxDisparity() - stereo_dataset.GetMinDisparity() + 15) & -16;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(stereo_dataset.GetMinDisparity(), disparity_num, 5, 8 * 3 * window_size * window_size,
                                                            32 * 3 * window_size * window_size, 2, 5, 5, 5, 0);

    cv::Ptr<cv::StereoMatcher> stereo2 = cv::ximgproc::createRightMatcher(stereo);

    double lamb = 8000.0;
    double sig = 1.5;

    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
    wls_filter->setLambda(lamb);
    wls_filter->setSigmaColor(sig);

    cv::Mat disparity;
    stereo->compute(rectified_images_[0], rectified_images_[1], disparity);

    cv::Mat disparity2;
    stereo2->compute(rectified_images_[1], rectified_images_[0], disparity2);

    disparity2.convertTo(disparity2, CV_32F);

    cv::Mat filteredImg;
    wls_filter->filter(disparity, stereo_dataset.GetImages()[0], filteredImg, disparity2);
    cv::threshold(filteredImg, filteredImg, 0, stereo_dataset.GetMaxDisparity() * 16, cv::THRESH_TOZERO);
    filteredImg.convertTo(filteredImg, CV_8U, 1.0/16.0);

    disparity_map_ = filteredImg;

}









