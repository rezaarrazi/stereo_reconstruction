#ifndef _CAMERA_POSE_ESTIMATOR
#define _CAMERA_POSE_ESTIMATOR


#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>


class CameraPoseEstimator
{

    private:
        std::array<cv::Mat, 2> camera_intrinsics_;
        std::array<std::vector<cv::Point2f>, 2> matched_points_;
        cv::Mat essential_matrix_;
        cv::Mat rotation_;
        cv::Mat translation_;

    public:
        CameraPoseEstimator() {}

        cv::Mat GetRotation() const;

        cv::Mat GetTranslation() const;

        void SetCameraIntrinsics(const std::array<cv::Mat, 2>& camera_intrinsics);

        void SetMatchedPoints(const std::array<std::vector<cv::Point2f>, 2>& matched_points);

        void EstimateCameraPose();

};


#endif






