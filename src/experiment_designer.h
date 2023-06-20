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

        cv::Mat rotation_gt_ = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        cv::Mat translation_gt_ = (cv::Mat_<float>(3, 1) << -1.0, 0.0, 0.0);
        std::array<std::string, 4> feature_extractor_names_ = {"ORB", "SIFT", "SURF", "BRISK"};

        std::array<float, 2> ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const;

    public:
        ExperimentDesigner() {}

        void CompareKeypointNumber();

        void CompareCameraPoseEstimation();

};






