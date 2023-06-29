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


void CameraPoseEstimator::EstimateCameraPose(std::size_t type)
{

    if (type == 0)
    {
        // Five-Point Algorithm
        essential_matrix_ = cv::findEssentialMat(matched_points_[0], matched_points_[1], camera_intrinsics_[0], cv::noArray(),
                                                 camera_intrinsics_[1], cv::noArray());
        cv::recoverPose(essential_matrix_, matched_points_[0], matched_points_[1], rotation_, translation_);
    }
    else if (type == 1)
    {
        // Seven-Point Algorithm
        cv::Mat fundamental_matrix = cv::findFundamentalMat(matched_points_[0], matched_points_[1], cv::FM_7POINT);
        essential_matrix_ = camera_intrinsics_[1].t() * fundamental_matrix * camera_intrinsics_[0];
        cv::recoverPose(essential_matrix_, matched_points_[0], matched_points_[1], rotation_, translation_);
    }
    else if (type == 2)
    {
        // Eight-Point Algorithm
        cv::Mat fundamental_matrix = cv::findFundamentalMat(matched_points_[0], matched_points_[1], cv::FM_8POINT);
        essential_matrix_ = camera_intrinsics_[1].t() * fundamental_matrix * camera_intrinsics_[0];
        cv::recoverPose(essential_matrix_, matched_points_[0], matched_points_[1], rotation_, translation_);
    }
    else if (type == 3)
        // Five-Point Algorithm
        cv::recoverPose(matched_points_[0], matched_points_[1], camera_intrinsics_[0], cv::noArray(),
                        camera_intrinsics_[1], cv::noArray(), essential_matrix_, rotation_, translation_);
    else
        std::cout << "type should be >= 0 and < 4\n";

}








