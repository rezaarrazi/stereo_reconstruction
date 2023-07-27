#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace torch;
using namespace torch::indexing;

class SuperGlue {
private:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const IValue &result);
    torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value);

    std::unique_ptr<torch::Device> device; // Pointer to torch::Device
    torch::jit::script::Module superpoint;
    torch::jit::script::Module superglue;
    

    std::array<cv::Mat, 2> image_pair; //images_pair[0]:image0; images_pair[1]:image1
    std::array<Tensor, 2> image_pair_tensor;

    std::array<std::vector<cv::Point2f>, 2> keypoints; //keypoints[0]:keypoints0; keypoints[1]:keypoints1
    std::array<Tensor, 2> keypoints_tensor;
    std::array<Tensor, 2> keypoints_scores_tensor;
    std::array<Tensor, 2> keypoints_descriptors_tensor;

    std::array<cv::Mat, 2> features; //features[0]:descriptors0; features[1]:descriptors1
    std::array<std::vector<cv::Point2f>, 2> matched_points; //features[0]:mkpts0; features[1]:mkpts1

public:
    SuperGlue(const std::string& superpointModelPath, const std::string& superglueModelPath);
    void SetImages(const std::string& image0Path, const std::string& image1Path, int downscaledWidth);
    void ExtractFeatures();
    void MatchFeatures(bool plot);
    
    // Add additional member functions as needed
    std::array<std::vector<cv::Point2f>, 2> GetMatchedPoints() const;
    const std::array<std::vector<cv::Point2f>, 2>& GetKeypoints() const;
    const std::array<cv::Mat, 2>& GetImagePair() const;
    std::size_t GetAverageKeypointNumber() const;

};
