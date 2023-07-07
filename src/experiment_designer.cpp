#include "experiment_designer.h"


std::array<double, 2> ExperimentDesigner::ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const
{

    std::array<double, 2> rmses = {0.0, 0.0};

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 3; j++)
            rmses[0] += pow(rotation.at<double>(i, j) - ROTATION_GT_.at<double>(i, j), 2);

    rmses[0] /= 9.0;
    rmses[0] = sqrt(rmses[0]);

    for (std::size_t i = 0; i < 3; i++)
        rmses[1] += pow(translation.at<double>(i, 0) - TRANSLATION_GT_.at<double>(i, 0), 2);

    rmses[1] /= 3.0;
    rmses[1] = sqrt(rmses[1]);

    return rmses;

}


double ExperimentDesigner::ComputePixelRatio(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt, double delta_d) const
{

    std::size_t total_pixel_count = 0;
    std::size_t count = 0;

    std::size_t delta_value = static_cast<std::size_t>(delta_d);

    uchar value = 0;
    uchar value_gt = 0;

    for (std::size_t i = 0; i < disparity_map.rows; i++)
    {
        for (std::size_t j = 0; j < disparity_map.cols; j++)
        {
            value = disparity_map.at<uchar>(i, j);
            value_gt = disparity_map_gt.at<uchar>(i, j);

            if ((value != 0) && (value_gt != 0))
            {
                total_pixel_count++;

                if ((value > value_gt) && (value - value_gt > static_cast<uchar>(delta_value)))
                    count++;
                else if ((value < value_gt) && (value_gt - value > static_cast<uchar>(delta_value)))
                    count++;
            }
        }
    }

    return static_cast<double>(count) / static_cast<double>(total_pixel_count);

}


double ExperimentDesigner::ComputeAverageError(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt) const
{

    std::size_t total_pixel_count = 0;
    std::size_t error = 0;

    uchar value = 0;
    uchar value_gt = 0;

    for (std::size_t i = 0; i < disparity_map.rows; i++)
    {
        for (std::size_t j = 0; j < disparity_map.cols; j++)
        {
            value = disparity_map.at<uchar>(i, j);
            value_gt = disparity_map_gt.at<uchar>(i, j);

            if ((value != 0) && (value_gt != 0))
            {
                total_pixel_count++;

                if (value > value_gt)
                    error += static_cast<std::size_t>(value - value_gt);
                else if (value < value_gt)
                    error += static_cast<std::size_t>(value_gt - value);
            }
        }
    }

    return static_cast<double>(error) / static_cast<double>(total_pixel_count);

}


double ExperimentDesigner::ComputeDisparityMapRMSE(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt) const
{

    std::size_t total_pixel_count = 0;
    std::size_t error = 0;

    uchar value = 0;
    uchar value_gt = 0;

    for (std::size_t i = 0; i < disparity_map.rows; i++)
    {
        for (std::size_t j = 0; j < disparity_map.cols; j++)
        {
            value = disparity_map.at<uchar>(i, j);
            value_gt = disparity_map_gt.at<uchar>(i, j);

            if ((value != 0) && (value_gt != 0))
            {
                total_pixel_count++;

                if (value > value_gt)
                    error += static_cast<std::size_t>(value - value_gt) * static_cast<std::size_t>(value - value_gt);
                else if (value < value_gt)
                    error += static_cast<std::size_t>(value_gt - value) * static_cast<std::size_t>(value_gt - value);
            }
        }
    }

    return sqrt(static_cast<double>(error) / static_cast<double>(total_pixel_count));

}


void ExperimentDesigner::CompareKeypointNumber()
{

    // The order is ORB, SIFT, SURF, BRISK
    std::array<std::size_t, 4> keypoint_numbers = {0, 0, 0, 0};

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        for (std::size_t j = 0; j < 4; j++)
        {
            feature_extractor_.ExtractFeatures(j);
            keypoint_numbers[j] += feature_extractor_.GetAverageKeypointNumber();
        }
    }

    for (std::size_t i = 0; i < 4; i++)
        std::cout << FEATURE_EXTRACTOR_NAMES_[i] << ": " << static_cast<std::size_t>(keypoint_numbers[i] / image_pair_number) << '\n';

}


void ExperimentDesigner::CompareFeatureExtractionAndMatching(std::size_t type, std::size_t keypoint_number, double distance_ratio, double ratio)
{

    // The order is ORB, SIFT, SURF, BRISK
    std::array<std::array<double, 2>, 4> rmses;

    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        for (std::size_t j = 0; j < 4; j++)
        {
            feature_extractor_.ExtractFeatures(j);

            if (type == 0)
                sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), keypoint_number);
            else if (type == 1)
                sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), distance_ratio);
            else if (type == 2)
                sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), ratio);
            
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose(3);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }

    }

    for (std::size_t i = 0; i < 4; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    std::cout << SPARSE_MATCHER_NAMES_[type] << '\n';

    for (std::size_t i = 0; i < 4; i++)
        std::cout << FEATURE_EXTRACTOR_NAMES_[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::CompareCameraPoseEstimation()
{

    std::array<std::array<double, 2>, 4> rmses;

    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;
    
    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        feature_extractor_.ExtractFeatures(3);
        sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 0.6);

        camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());

        for (std::size_t j = 0; j < 4; j++)
        {
            camera_pose_estimator_.EstimateCameraPose(j);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }
    }

    for (std::size_t i = 0; i < 4; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    for (std::size_t i = 0; i < 4; i++)
        std::cout << CAMERA_POSE_ESTIMATOR_NAMES_[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::PrintDisparityMaps(std::size_t index)
{

    stereo_dataset_.SetImages(index);
    stereo_dataset_.SetCalibrations(index);
    stereo_dataset_.SetDisparityMaps(index);

    feature_extractor_.SetImages(stereo_dataset_.GetImages());

    std::array<std::size_t, 2> feature_extractor_types = {1, 3};
    std::array<std::string, 2> feature_extractor_names = {"SIFT", "BRISK"};
    std::array<std::string, 2> dense_matcher_names = {"StereoBM", "StereoSGBM"};

    cv::Mat rotation;
    cv::Mat translation;

    for (std::size_t i = 0; i < 2; i++)
    {
        feature_extractor_.ExtractFeatures(feature_extractor_types[i]);

        if (feature_extractor_types[i] == 1)
            sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 50);
        else
            sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 0.6);
        
        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());

        camera_pose_estimator_.EstimateCameraPose(3);

        rotation = camera_pose_estimator_.GetRotation();
        translation = camera_pose_estimator_.GetTranslation();

        dense_matcher_.LoadData(stereo_dataset_, rotation, translation);

        dense_matcher_.RectifyImages();

        for (std::size_t j = 0; j < 2; j++)
        {
            dense_matcher_.ComputeDisparityMap(j);
            cv::imwrite("../" + feature_extractor_names[i] + dense_matcher_names[j] + ".png", dense_matcher_.GetColorfulDisparityMap());
        }
    }

    cv::Mat gt;
    cv::applyColorMap(stereo_dataset_.GetDisparityMaps()[0], gt, cv::COLORMAP_JET);
    cv::imwrite("../gt.png", gt);

}


void ExperimentDesigner::CompareDisparityMaps(std::size_t feature_extractor_type, std::size_t sparse_matcher_type, std::size_t dense_matcher_type)
{

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    double pixel_ratio_1 = 0.0;
    double pixel_ratio_2 = 0.0;
    double pixel_ratio_4 = 0.0;
    double average_error = 0.0;
    double rmse = 0.0;

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);
        stereo_dataset_.SetDisparityMaps(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        feature_extractor_.ExtractFeatures(feature_extractor_type);

        if (sparse_matcher_type == 0)
            sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 50);
        else if (sparse_matcher_type == 1)
            sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 3.0);
        else
            sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 0.6);
        
        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());

        camera_pose_estimator_.EstimateCameraPose(3);

        dense_matcher_.LoadData(stereo_dataset_, camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

        dense_matcher_.RectifyImages();

        dense_matcher_.ComputeDisparityMap(dense_matcher_type);

        pixel_ratio_1 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 1.0);
        pixel_ratio_2 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 2.0);
        pixel_ratio_4 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 4.0);
        average_error += ComputeAverageError(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0]);
        rmse += ComputeDisparityMapRMSE(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0]);
    }

    std::cout << "feature extractor: " << FEATURE_EXTRACTOR_NAMES_[feature_extractor_type] << '\n';
    std::cout << "sparse matcher: " << SPARSE_MATCHER_NAMES_[sparse_matcher_type] << '\n';
    std::cout << "dense matcher: " << DENSE_MATCHER_NAMES_[dense_matcher_type] << '\n';
    std::cout << "pixel ratio 1.0: " << pixel_ratio_1 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "pixel ratio 2.0: " << pixel_ratio_2 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "pixel ratio 4.0: " << pixel_ratio_4 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "average error: " << average_error / static_cast<double>(image_pair_number) << '\n';
    std::cout << "rmse: " << rmse / static_cast<double>(image_pair_number) << '\n';

}








