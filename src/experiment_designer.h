#ifndef _EXPERIMENT_DESIGNER
#define _EXPERIMENT_DESIGNER


#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"
#include "camera_pose_estimator.h"
#include "dense_matcher.h"
#include "scene_reconstructor.h"


class ExperimentDesigner
{

    private:
        StereoDataset stereo_dataset_;
        FeatureExtractor feature_extractor_;
        SparseMatcher sparse_matcher_;
        CameraPoseEstimator camera_pose_estimator_;
        DenseMatcher dense_matcher_;
        SceneReconstructor scene_reconstructor_;

        cv::Mat ROTATION_GT_ = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        cv::Mat TRANSLATION_GT_ = (cv::Mat_<double>(3, 1) << -1.0, 0.0, 0.0);

        std::array<std::string, 4> FEATURE_EXTRACTOR_NAMES_ = {"ORB", "SIFT", "SURF", "BRISK"};
        std::array<std::string, 3> SPARSE_MATCHER_NAMES_ = {"BFSortTop", "BFMinDistance", "FLANNBased"};
        std::array<std::string, 4> CAMERA_POSE_ESTIMATOR_NAMES_ = {"Five-Point Algorithm 1", "Seven-Point Algorithm",
                                                                   "Eight-Point Algorithm", "Five-Point Algorithm 2"};
        std::array<std::string, 2> DENSE_MATCHER_NAMES_ = {"StereoBM", "StereoSGBM"};

        std::array<double, 2> ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const;

        double ComputePixelRatio(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt, double delta_d) const;
        double ComputeAverageError(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt) const;
        double ComputeDisparityMapRMSE(const cv::Mat& disparity_map, const cv::Mat& disparity_map_gt) const;

    public:
        ExperimentDesigner() {}

        void CompareKeypointNumber();

        void CompareFeatureExtractionAndBFSortTop(std::size_t feature_extractor_type);

        void CompareFeatureExtractionAndBFMinDistance(std::size_t feature_extractor_type);

        void CompareFeatureExtractionAndFLANNBased(std::size_t feature_extractor_type);
        
        void CompareCameraPoseEstimation();

        void PrintMatchedImages();

        void CompareDisparityMaps(std::size_t dense_matcher_type);

        void PrintDisparityMaps(std::size_t index);

        void ReconstructScenesDirectly(std::size_t index, std::size_t dense_matcher_type);

        void ReconstructScenesGT(std::size_t index);

        void ReconstructScenes(std::size_t index, std::size_t dense_matcher_type, const std::string& mesh_index);

        void ReconstructScenesGT1(std::size_t index, const std::string& mesh_index);

};


#endif






