#include "dense_matcher.h"


void DenseMatcher::FillHolesMaximum(const StereoDataset& stereo_dataset, std::size_t window_size)
{

    std::size_t image_height = stereo_dataset.GetImageHeight();
    std::size_t image_width = stereo_dataset.GetImageWidth();
    uchar maximum_value = 0;

    cv::Mat filled_disparity_map = disparity_map_.clone();

    std::size_t half_size = static_cast<std::size_t>(window_size / 2);

    for (std::size_t i = half_size; i < image_height - half_size; i++)
        for (std::size_t j = half_size; j < image_width - half_size; j++)
            if (disparity_map_.at<uchar>(i, j) < 10)
            {
                maximum_value = 0;

                for (std::size_t k = i - half_size; k < i + half_size + 1; k++)
                    for (std::size_t l = j - half_size; l < j + half_size + 1; l++)
                        if (disparity_map_.at<uchar>(k, l) > maximum_value)
                            maximum_value = disparity_map_.at<uchar>(k, l);

                filled_disparity_map.at<uchar>(i, j) = maximum_value;
            }

    disparity_map_ = filled_disparity_map;

}


void DenseMatcher::FillHolesAverage(const StereoDataset& stereo_dataset, std::size_t window_size)
{

    std::size_t image_height = stereo_dataset.GetImageHeight();
    std::size_t image_width = stereo_dataset.GetImageWidth();
    std::size_t value = 0;

    cv::Mat filled_disparity_map = disparity_map_.clone();

    std::size_t half_size = static_cast<std::size_t>(window_size / 2);

    std::size_t window_element_number = window_size * window_size;

    for (std::size_t i = half_size; i < image_height - half_size; i++)
        for (std::size_t j = half_size; j < image_width - half_size; j++)
            if (disparity_map_.at<uchar>(i, j) < 10)
            {
                value = 0;

                for (std::size_t k = i - half_size; k < i + half_size + 1; k++)
                    for (std::size_t l = j - half_size; l < j + half_size + 1; l++)
                        value += static_cast<std::size_t>(disparity_map_.at<uchar>(k, l));
                
                filled_disparity_map.at<uchar>(i, j) = static_cast<uchar>(static_cast<std::size_t>(value / window_element_number));
            }

    disparity_map_ = filled_disparity_map;

}


cv::Mat DenseMatcher::GetDisparityMap() const
{
    return disparity_map_;
}


cv::Mat DenseMatcher::GetColorfulDisparityMap() const
{
    return colorful_disparity_map_;
}


void DenseMatcher::RectifyImages(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation)
{

    std::array<cv::Mat, 2> images = stereo_dataset.GetImages();

    cv::Size image_size = stereo_dataset.GetImageSize();

    std::array<cv::Mat, 2> camera_intrinsics = stereo_dataset.GetCameraIntrinsics();

    // rectify images
    cv::Mat r0, r1, p0, p1, q;
    cv::stereoRectify(camera_intrinsics[0], cv::noArray(), camera_intrinsics[1], cv::noArray(),
                      image_size, rotation, translation, r0, r1, p0, p1, q);
    
    cv::Mat map00, map01, map10, map11;
    cv::initUndistortRectifyMap(camera_intrinsics[0], cv::noArray(), r0, p0, image_size, CV_32F, map00, map01);
    cv::initUndistortRectifyMap(camera_intrinsics[1], cv::noArray(), r1, p1, image_size, CV_32F, map10, map11);

    cv::remap(images[0], rectified_images_[0], map00, map01, cv::INTER_CUBIC);
    cv::remap(images[1], rectified_images_[1], map10, map11, cv::INTER_CUBIC);

}


void DenseMatcher::ComputeDisparityMap(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation, std::size_t type)
{

    if (type == 0)
    {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(bm_disparity_number_, bm_block_size_);
        bm->setPreFilterCap(pre_filter_cap_);
        bm->setUniquenessRatio(uniqueness_ratio_);
        bm->compute(rectified_images_[0], rectified_images_[1], disparity_map_);
        // disparity_map_.convertTo(disparity_map_, CV_8U, 255.0 / (16.0 * bm_disparity_number_));
        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);
    }
    else if (type == 1)
    {
        int sgbm_p1 = 8 * 3 * sgbm_block_size_ * sgbm_block_size_;
        int sgbm_p2 = 32 * 3 * sgbm_block_size_ * sgbm_block_size_;
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(sgbm_min_disparity_, sgbm_disparity_number_, sgbm_block_size_,
                                                              sgbm_p1, sgbm_p2, sgbm_disp12_max_diff_, pre_filter_cap_,
                                                              uniqueness_ratio_, sgbm_speckle_window_size_, sgbm_speckle_range_);
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

        // cv::Ptr<cv::StereoMatcher> sgbm1 = cv::ximgproc::createRightMatcher(sgbm);

        // float lambd = 8000.0;
        // float sigma = 2.0;

        // cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
        // wls_filter->setLambda(lambd);
        // wls_filter->setSigmaColor(sigma);

        sgbm->compute(rectified_images_[0], rectified_images_[1], disparity_map_);
        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);
        // FillHolesMaximum(stereo_dataset, 5);
        FillHolesAverage(stereo_dataset, 10);
        

        // cv::Mat disparity_map1;
        // sgbm1->compute(rectified_images_[1], rectified_images_[0], disparity_map1);

        // disparity_map1.convertTo(disparity_map1, CV_32F);

        // cv::Mat filtered_disparity_map;
        // wls_filter->filter(disparity_map_, stereo_dataset.GetImages()[0], filtered_disparity_map, disparity_map1);

        //cv::threshold(filtered_disparity_map, filtered_disparity_map, 0, stereo_dataset.GetMaxDisparity() * 16, cv::THRESH_TOZERO);
        // filtered_disparity_map.convertTo(filtered_disparity_map, CV_8U, 1.0 / 16.0);

        // disparity_map_ = filtered_disparity_map;

        cv::applyColorMap(disparity_map_, colorful_disparity_map_, cv::COLORMAP_JET);




        



        

        
    }
    else
        std::cout << "type should be >= 0 and < 2.\n";

    
}







