#include "disparity.h"


cv::Mat Disparity::GetDisparity() const
{
    return disparity_;
}

void Disparity::ComputeDisparity()
{
    std::array<cv::Mat, 2> camera_intrinsics_ = dataset_.GetCameraIntrinsics();

    // Rectify images
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::stereoRectify(camera_intrinsics_[0], cv::noArray(), camera_intrinsics_[1], cv::noArray(), dataset_.GetImageSize(), R_, t_, R1_, R2_, P1_, P2_, Q_);
    
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(camera_intrinsics_[1], cv::noArray(), R1_, P1_, dataset_.GetImageSize(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(camera_intrinsics_[0], cv::noArray(), R2_, P2_, dataset_.GetImageSize(), CV_16SC2, map21, map22);

    cv::remap(dataset_.GetImages()[0], image_left_rec_, map11, map12, cv::INTER_CUBIC);
    cv::remap(dataset_.GetImages()[1], image_right_rec_, map21, map22, cv::INTER_CUBIC);
    
    // Create a StereoSGBM object
    int window_size = 5;
    int num_disparities = ((dataset_.GetMaxDisparity()-dataset_.GetMinDisparity())+15) & -16;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(dataset_.GetMinDisparity(),
                                                            num_disparities,
                                                            5,
                                                            8*3*window_size*window_size,
                                                            32*3*window_size*window_size,
                                                            2,
                                                            5,
                                                            5,
                                                            5,
                                                            0);

    cv::Ptr<cv::StereoMatcher> stereo2 = cv::ximgproc::createRightMatcher(stereo);

    double lamb = 8000.0;
    double sig = 1.5;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
    wls_filter->setLambda(lamb);
    wls_filter->setSigmaColor(sig);

    cv::Mat disparity;
    stereo->compute(image_left_rec_, image_right_rec_, disparity);

    cv::Mat disparity2;
    stereo2->compute(image_right_rec_, image_left_rec_, disparity2);

    disparity2.convertTo(disparity2, CV_32F);

    cv::Mat filteredImg;
    wls_filter->filter(disparity, dataset_.GetImages()[0], filteredImg, disparity2);
    cv::threshold(filteredImg, filteredImg, 0, dataset_.GetMaxDisparity() * 16, cv::THRESH_TOZERO);
    filteredImg.convertTo(filteredImg, CV_8U, 1.0/16.0);

    disparity_ = filteredImg;
}