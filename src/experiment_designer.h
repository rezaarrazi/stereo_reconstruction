#ifndef _EXPERIMENT_DESIGNER
#define _EXPERIMENT_DESIGNER


#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"
#include "camera_pose_estimator.h"


class ExperimentDesigner
{

    private:
        StereoDataset stereo_dataset_;
        FeatureExtractor feature_extractor_;
        SparseMatcher sparse_matcher_;
        CameraPoseEstimator camera_pose_estimator_;

        cv::Mat rotation_gt_ = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        cv::Mat translation_gt_ = (cv::Mat_<double>(3, 1) << -1.0, 0.0, 0.0);
        std::array<std::string, 4> feature_extractor_names_ = {"ORB", "SIFT", "SURF", "BRISK"};
        std::array<std::string, 3> sparse_matcher_names_ = {"BFSortTop", "BFMinDistance", "FLANNBased"};
        std::array<std::string, 4> camera_pose_estimator_names_ = {"Five-Point Algorithm 1", "Seven-Point Algorithm",
                                                                   "Eight-Point Algorithm", "Five-Point Algorithm 2"};

        std::array<double, 2> ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const;

    public:
        ExperimentDesigner() {}

        void CompareKeypointNumber();

        void CompareFeatureExtractionAndMatching(std::size_t type, std::size_t keypoint_number = 50,
                                                 double distance_ratio = 3.0, double ratio = 0.6);
        
        void CompareCameraPoseEstimation();

};


#endif






