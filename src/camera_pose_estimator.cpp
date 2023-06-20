#include "camera_pose_estimator.h"


cv::Mat CameraPoseEstimator::GetRotation() const
{
    return rotation_;
}


cv::Mat CameraPoseEstimator::GetTranslation() const
{
    return translation_;
}


void CameraPoseEstimator::SetCameraIntrinsics(const std::array<cv::Mat, 2>& camera_intrinsics)
{
    camera_intrinsics_ = camera_intrinsics;
}


void CameraPoseEstimator::SetMatchedPoints(const std::array<std::vector<cv::Point2f>, 2>& matched_points)
{
    matched_points_ = matched_points;
}


void CameraPoseEstimator::EstimateCameraPose()
{
    cv::recoverPose(matched_points_[0], matched_points_[1],
                    camera_intrinsics_[0], cv::noArray(), camera_intrinsics_[1], cv::noArray(),
                    essential_matrix_, rotation_, translation_);
}








