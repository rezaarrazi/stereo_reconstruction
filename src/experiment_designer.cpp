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

    // The order is ORB, SIFT, SURF, BRISK, SUPERGLUE
    std::array<std::size_t, 5> keypoint_numbers = {0, 0, 0, 0, 0};

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

        superglue_.run(stereo_dataset_.GetImagePairPath()[0], stereo_dataset_.GetImagePairPath()[1], 640, true);
        keypoint_numbers[4] += superglue_.GetAverageKeypointNumber();
    }

    for (std::size_t i = 0; i < 5; i++)
        std::cout << FEATURE_EXTRACTOR_NAMES_[i] << ": " << static_cast<std::size_t>(keypoint_numbers[i] / image_pair_number) << '\n';

}


void ExperimentDesigner::CompareFeatureExtractionAndBFSortTop(std::size_t feature_extractor_type)
{

    std::array<std::size_t, 3> keypoint_numbers = {50, 100, 150};

    std::array<std::array<double, 2>, 3> rmses;

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        for (std::size_t j = 0; j < 3; j++)
        {
            feature_extractor_.ExtractFeatures(feature_extractor_type);

            sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), keypoint_numbers[j]);
            
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose(3);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }

    }

    for (std::size_t i = 0; i < 3; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    std::cout << FEATURE_EXTRACTOR_NAMES_[feature_extractor_type] << '\n';

    for (std::size_t i = 0; i < 3; i++)
        std::cout << keypoint_numbers[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::CompareFeatureExtractionAndBFMinDistance(std::size_t feature_extractor_type)
{

    std::array<double, 3> distance_ratios = {3.0, 4.0, 5.0};

    std::array<std::array<double, 2>, 3> rmses;

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        for (std::size_t j = 0; j < 3; j++)
        {
            feature_extractor_.ExtractFeatures(feature_extractor_type);

            sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), distance_ratios[j]);
            
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose(3);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }

    }

    for (std::size_t i = 0; i < 3; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    std::cout << FEATURE_EXTRACTOR_NAMES_[feature_extractor_type] << '\n';

    for (std::size_t i = 0; i < 3; i++)
        std::cout << distance_ratios[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::CompareFeatureExtractionAndFLANNBased(std::size_t feature_extractor_type)
{

    std::array<double, 3> ratios = {0.3, 0.6, 0.8};

    std::array<std::array<double, 2>, 3> rmses;

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        for (std::size_t j = 0; j < 3; j++)
        {
            feature_extractor_.ExtractFeatures(feature_extractor_type);

            sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), ratios[j]);
            
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose(3);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }

    }

    for (std::size_t i = 0; i < 3; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    std::cout << FEATURE_EXTRACTOR_NAMES_[feature_extractor_type] << '\n';

    for (std::size_t i = 0; i < 3; i++)
        std::cout << ratios[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}

void ExperimentDesigner::SuperGlueRotationTranslationError()
{
    std::array<double, 2> rmses;

    for (std::size_t j = 0; j < 2; j++)
        rmses[j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);

        superglue_.run(stereo_dataset_.GetImagePairPath()[0], stereo_dataset_.GetImagePairPath()[1], 640, true);

        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        camera_pose_estimator_.SetMatchedPoints(superglue_.GetMatchedPoints());
        camera_pose_estimator_.EstimateCameraPose(3);

        std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

        rmses[0] += rmse[0];
        rmses[1] += rmse[1];

    }

    rmses[0] /= image_pair_number;
    rmses[1] /= image_pair_number;

    std::cout << FEATURE_EXTRACTOR_NAMES_[4] << '\n';
    std::cout << ": rotation: " << rmses[0] << " translation: " << rmses[1] << '\n';

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

        feature_extractor_.ExtractFeatures(2);

        sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 5.0);

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


void ExperimentDesigner::PrintMatchedImages()
{

    stereo_dataset_.SetImages(0);

    feature_extractor_.SetImages(stereo_dataset_.GetImages());

    feature_extractor_.ExtractFeatures(2);

    sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 5.0);

    sparse_matcher_.DisplayMatchings(stereo_dataset_.GetImages(), feature_extractor_.GetKeypoints(), true);
}


void ExperimentDesigner::CompareDisparityMaps(std::size_t dense_matcher_type)
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

        dense_matcher_.LoadDataDirectly(stereo_dataset_);

        dense_matcher_.ComputeDisparityMapDirectly(dense_matcher_type);

        pixel_ratio_1 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 1.0);
        pixel_ratio_2 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 2.0);
        pixel_ratio_4 += ComputePixelRatio(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0], 4.0);
        average_error += ComputeAverageError(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0]);
        rmse += ComputeDisparityMapRMSE(dense_matcher_.GetDisparityMap(), stereo_dataset_.GetDisparityMaps()[0]);
    }

    std::cout << "dense matcher: " << DENSE_MATCHER_NAMES_[dense_matcher_type] << '\n';
    std::cout << "pixel ratio 1.0: " << pixel_ratio_1 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "pixel ratio 2.0: " << pixel_ratio_2 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "pixel ratio 4.0: " << pixel_ratio_4 / static_cast<double>(image_pair_number) << '\n';
    std::cout << "average error: " << average_error / static_cast<double>(image_pair_number) << '\n';
    std::cout << "rmse: " << rmse / static_cast<double>(image_pair_number) << '\n';

}


void ExperimentDesigner::PrintDisparityMaps(std::size_t index)
{

    stereo_dataset_.SetImages(index);
    stereo_dataset_.SetCalibrations(index);
    stereo_dataset_.SetDisparityMaps(index);

    dense_matcher_.LoadDataDirectly(stereo_dataset_);

    for (std::size_t j = 0; j < 2; j++)
    {
        dense_matcher_.ComputeDisparityMapDirectly(j);
        cv::imwrite("../2" + DENSE_MATCHER_NAMES_[j] + ".png", dense_matcher_.GetColorfulDisparityMap());
    }

    cv::Mat gt;
    cv::applyColorMap(stereo_dataset_.GetDisparityMaps()[0], gt, cv::COLORMAP_JET);
    cv::imwrite("../2gt.png", gt);

}


void ExperimentDesigner::ReconstructScenesDirectly(std::size_t index, std::size_t dense_matcher_type)
{

    stereo_dataset_.SetImages(index);

    stereo_dataset_.SetCalibrations(index);

    dense_matcher_.LoadDataDirectly(stereo_dataset_);

    dense_matcher_.ComputeDisparityMapDirectly(dense_matcher_type);

    stereo_dataset_.SetDisparityMaps(index);
    
    float distance_threshold = 20000.0;

    scene_reconstructor_.LoadData(stereo_dataset_);

    scene_reconstructor_.ReconstructScene(dense_matcher_.GetDisparityMap(), distance_threshold);

    scene_reconstructor_.WriteMeshToFile("../1" + DENSE_MATCHER_NAMES_[dense_matcher_type] + "mesh.off");

}


void ExperimentDesigner::ReconstructScenesGT(std::size_t index)
{

    stereo_dataset_.SetImages(index);

    stereo_dataset_.SetCalibrations(index);

    stereo_dataset_.SetDisparityMaps(index);
    
    float distance_threshold = 20000.0;

    scene_reconstructor_.LoadData(stereo_dataset_);

    scene_reconstructor_.ReconstructScene(stereo_dataset_.GetDisparityMaps()[0], distance_threshold);

    scene_reconstructor_.WriteMeshToFile("../1gtmesh.off");

}


void ExperimentDesigner::ReconstructScenes(std::size_t index, std::size_t dense_matcher_type, const std::string& mesh_index)
{

    stereo_dataset_.SetImages(index);

    stereo_dataset_.SetCalibrations(index);

    dense_matcher_.LoadDataDirectly(stereo_dataset_);

    dense_matcher_.ComputeDisparityMapWithoutConversion(dense_matcher_type);

    stereo_dataset_.SetDisparityMaps(index);
    
    float distance_threshold = 300.0;

    scene_reconstructor_.LoadData(stereo_dataset_);

    scene_reconstructor_.ReconstructSceneDirectly(dense_matcher_.GetDisparityMap(), stereo_dataset_, distance_threshold,
                                                  "../" + mesh_index + DENSE_MATCHER_NAMES_[dense_matcher_type] + "mesh.off");

}


void ExperimentDesigner::ReconstructScenesGT1(std::size_t index, const std::string& mesh_index)
{

    stereo_dataset_.SetImages(index);

    stereo_dataset_.SetCalibrations(index);

    stereo_dataset_.SetDisparityMaps(index);

    cv::Mat disparity_map = stereo_dataset_.GetDisparityMaps()[0];

    disparity_map.convertTo(disparity_map, CV_16SC1);
    
    float distance_threshold = 300.0;

    scene_reconstructor_.LoadData(stereo_dataset_);

    scene_reconstructor_.ReconstructSceneDirectly(disparity_map, stereo_dataset_, distance_threshold, "../" + mesh_index + "gtmesh.off");

}








