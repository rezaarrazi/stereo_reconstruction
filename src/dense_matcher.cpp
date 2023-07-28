#include "dense_matcher.h"


void DenseMatcher::LoadData(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation)
{
    images_[0] = stereo_dataset.GetImages()[0].clone();
    images_[1] = stereo_dataset.GetImages()[1].clone();
    
    image_size_ = stereo_dataset.GetImageSize();
    camera_intrinsics_ = stereo_dataset.GetCameraIntrinsics();
    rotation_ = rotation;
    translation_ = translation;
}


void DenseMatcher::LoadDataDirectly(const StereoDataset& stereo_dataset)
{
    images_[0] = stereo_dataset.GetImages()[0].clone();
    images_[1] = stereo_dataset.GetImages()[1].clone();

    image_size_ = stereo_dataset.GetImageSize();
    camera_intrinsics_ = stereo_dataset.GetCameraIntrinsics();
}


cv::Mat DenseMatcher::GetDisparityMap() const
{
    return disparity_map_;
}


cv::Mat DenseMatcher::GetColorfulDisparityMap() const
{
    return colorful_disparity_map_;
}


cv::Mat DenseMatcher::GetProjectionMatrix() const
{
    return projection_matrix_;
}


void DenseMatcher::RectifyImages()
{

    // rectify images
    cv::Mat r0, r1, p0, p1;
    cv::stereoRectify(camera_intrinsics_[0], cv::noArray(), camera_intrinsics_[1], cv::noArray(), image_size_,
                      rotation_, translation_, r0, r1, p0, p1, projection_matrix_, cv::CALIB_ZERO_DISPARITY);
    
    cv::Mat map00, map01, map10, map11;
    cv::initUndistortRectifyMap(camera_intrinsics_[0], cv::noArray(), r0, p0, image_size_, CV_32F, map00, map01);
    cv::initUndistortRectifyMap(camera_intrinsics_[1], cv::noArray(), r1, p1, image_size_, CV_32F, map10, map11);

    cv::remap(images_[0], rectified_images_[0], map00, map01, cv::INTER_CUBIC);
    cv::remap(images_[1], rectified_images_[1], map10, map11, cv::INTER_CUBIC);

}


void DenseMatcher::ComputeDisparityMap(std::size_t type)
{

    if (type == 0)
    {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(BM_DISPARITY_NUMBER_, BM_BLOCK_SIZE_);
        bm->setPreFilterCap(PRE_FILTER_CAP_);
        bm->setUniquenessRatio(UNIQUENESS_RATIO_);

        std::array<cv::Mat, 2> grayscale_images;
        cv::cvtColor(rectified_images_[0], grayscale_images[0], CV_BGR2GRAY);
        cv::cvtColor(rectified_images_[1], grayscale_images[1], CV_BGR2GRAY);

        bm->compute(grayscale_images[0], grayscale_images[1], disparity_map_);

        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);

        cv::applyColorMap(disparity_map_, colorful_disparity_map_, cv::COLORMAP_JET);
    }
    else if (type == 1)
    {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(SGBM_MIN_DISPARITY_, SGBM_DISPARITY_NUMBER_, SGBM_BLOCK_SIZE_,
                                                              8 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_, 32 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_,
                                                              SGBM_DISP12_MAX_DIFF_, PRE_FILTER_CAP_, UNIQUENESS_RATIO_,
                                                              SGBM_SPECKLE_WINDOW_SIZE_, SGBM_SPECKLE_RANGE_, cv::StereoSGBM::MODE_SGBM);

        // cv::Ptr<cv::StereoMatcher> sgbm1 = cv::ximgproc::createRightMatcher(sgbm);

        // float lambd = 8000.0;
        // float sigma = 2.0;

        // cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
        // wls_filter->setLambda(lambd);
        // wls_filter->setSigmaColor(sigma);

        sgbm->compute(rectified_images_[0], rectified_images_[1], disparity_map_);
        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);

        // FillHoles(1, 15);

        // cv::Mat disparity_map1;
        // sgbm1->compute(rectified_images_[1], rectified_images_[0], disparity_map1);

        // disparity_map1.convertTo(disparity_map1, CV_32F);

        // cv::Mat filtered_disparity_map;
        // wls_filter->filter(disparity_map_, images_[0], filtered_disparity_map, disparity_map1);

        // cv::threshold(filtered_disparity_map, filtered_disparity_map, 0, stereo_dataset.GetMaxDisparity() * 16, cv::THRESH_TOZERO);
        // filtered_disparity_map.convertTo(filtered_disparity_map, CV_8U, 1.0 / 16.0);

        // disparity_map_ = filtered_disparity_map;

        cv::applyColorMap(disparity_map_, colorful_disparity_map_, cv::COLORMAP_JET);

    }
    else
        std::cout << "type should be >= 0 and < 2.\n";
    
}


void DenseMatcher::ComputeDisparityMapDirectly(std::size_t type)
{

    if (type == 0)
    {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(BM_DISPARITY_NUMBER_, BM_BLOCK_SIZE_);
        bm->setPreFilterCap(PRE_FILTER_CAP_);
        bm->setUniquenessRatio(UNIQUENESS_RATIO_);

        std::array<cv::Mat, 2> grayscale_images;
        cv::cvtColor(images_[0], grayscale_images[0], CV_BGR2GRAY);
        cv::cvtColor(images_[1], grayscale_images[1], CV_BGR2GRAY);

        bm->compute(grayscale_images[0], grayscale_images[1], disparity_map_);

        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);

        cv::applyColorMap(disparity_map_, colorful_disparity_map_, cv::COLORMAP_JET);
    }
    else if (type == 1)
    {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(SGBM_MIN_DISPARITY_, SGBM_DISPARITY_NUMBER_, SGBM_BLOCK_SIZE_,
                                                              8 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_, 32 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_,
                                                              SGBM_DISP12_MAX_DIFF_, PRE_FILTER_CAP_, UNIQUENESS_RATIO_,
                                                              SGBM_SPECKLE_WINDOW_SIZE_, SGBM_SPECKLE_RANGE_, cv::StereoSGBM::MODE_SGBM);

        sgbm->compute(images_[0], images_[1], disparity_map_);
        disparity_map_.convertTo(disparity_map_, CV_8U, 1.0 / 16.0);
        cv::applyColorMap(disparity_map_, colorful_disparity_map_, cv::COLORMAP_JET);

    }
    else
        std::cout << "type should be >= 0 and < 2.\n";

}


void DenseMatcher::ComputeDisparityMapWithoutConversion(std::size_t type)
{
    if (type == 0)
    {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(BM_DISPARITY_NUMBER_, BM_BLOCK_SIZE_);
        bm->setPreFilterCap(PRE_FILTER_CAP_);
        bm->setUniquenessRatio(UNIQUENESS_RATIO_);

        std::array<cv::Mat, 2> grayscale_images;
        cv::cvtColor(images_[0], grayscale_images[0], CV_BGR2GRAY);
        cv::cvtColor(images_[1], grayscale_images[1], CV_BGR2GRAY);

        bm->compute(grayscale_images[0], grayscale_images[1], disparity_map_);
    }
    else if (type == 1)
    {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(SGBM_MIN_DISPARITY_, SGBM_DISPARITY_NUMBER_, SGBM_BLOCK_SIZE_,
                                                              8 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_, 32 * 3 * SGBM_BLOCK_SIZE_ * SGBM_BLOCK_SIZE_,
                                                              SGBM_DISP12_MAX_DIFF_, PRE_FILTER_CAP_, UNIQUENESS_RATIO_,
                                                              SGBM_SPECKLE_WINDOW_SIZE_, SGBM_SPECKLE_RANGE_, cv::StereoSGBM::MODE_SGBM);

        sgbm->compute(images_[0], images_[1], disparity_map_);
    }
    else
        std::cout << "type should be >= 0 and < 2.\n";
}


void DenseMatcher::FilterMedian()
{

    std::size_t rows = disparity_map_.rows;
    std::size_t cols = disparity_map_.cols;

    std::size_t row_index = static_cast<std::size_t>(rows / 2);
    std::size_t col_index = 0;

    while (disparity_map_.at<uchar>(row_index, col_index) == 0)
        col_index++;

    cv::Mat filtered_disparity_map = disparity_map_.clone();

    std::array<uchar, 9> values = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t index = 0;

    for (std::size_t i = 1; i < rows - 1; i++)
    {
        for (std::size_t j = col_index; j < cols - 1; j++)
        {
            if (disparity_map_.at<uchar>(i, j) == 0)
            {
                index = 0;

                for (std::size_t k = i - 1; k < i + 2; k++)
                    for (std::size_t l = j - 1; l < j + 2; l++)
                        values[index++] = disparity_map_.at<uchar>(i, j);

                std::sort(values.begin(), values.end());

                filtered_disparity_map.at<uchar>(i, j) = values[4];
            }
        }
    }

    disparity_map_ = filtered_disparity_map;

}







