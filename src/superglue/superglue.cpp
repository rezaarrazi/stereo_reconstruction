#include "superglue.h"
#include "io.h"
#include "viz.h"
#include <filesystem>

namespace fs = std::filesystem;

SuperGlue::SuperGlue(const std::string& superpointModelPath, const std::string& superglueModelPath) {
    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    device = std::make_unique<torch::Device>(torch::kCPU);
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = std::make_unique<torch::Device>(torch::kCUDA);
    }
    
    // Look for the TorchScript module files
    if (!fs::exists(superpointModelPath)) {
      std::cerr << "Could not find the TorchScript module file " << superpointModelPath << std::endl;
    }
    superpoint = torch::jit::load(superpointModelPath);
    superpoint.eval();
    superpoint.to(*device);
    
    if (!fs::exists(superglueModelPath)) {
      std::cerr << "Could not find the TorchScript module file " << superglueModelPath << std::endl;
    }
    superglue = torch::jit::load(superglueModelPath);
    superglue.eval();
    superglue.to(*device);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SuperGlue::unpack_result(const IValue &result) {
    auto dict = result.toGenericDict();
    return {dict.at("keypoints").toTensorVector()[0], //
            dict.at("scores").toTensorVector()[0],    //
            dict.at("descriptors").toTensorVector()[0]};
}

torch::Dict<std::string, Tensor> SuperGlue::toTensorDict(const torch::IValue &value) {
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}

void SuperGlue::SetImages(const std::string& image0Path, const std::string& image1Path, int downscaledWidth) {
    image_pair_tensor[0] = read_image(image0Path, downscaledWidth).to(*device);
    image_pair_tensor[1] = read_image(image1Path, downscaledWidth).to(*device);
}

void SuperGlue::ExtractFeatures() {
    // Call SuperPoint model
    std::tie(keypoints_tensor[0], keypoints_scores_tensor[0], keypoints_descriptors_tensor[0]) = unpack_result(superpoint.forward({image_pair_tensor[0]}));
    std::tie(keypoints_tensor[1], keypoints_scores_tensor[1], keypoints_descriptors_tensor[1]) = unpack_result(superpoint.forward({image_pair_tensor[1]}));

    auto keypoints0Accessor = keypoints_tensor[0].accessor<float, 2>();
    for (int i = 0; i < keypoints_tensor[0].size(0); ++i) {
        keypoints[0].push_back(cv::Point2f(keypoints0Accessor[i][0], keypoints0Accessor[i][1]));
    }

    auto keypoints1Accessor = keypoints_tensor[1].accessor<float, 2>();
    for (int i = 0; i < keypoints_tensor[1].size(0); ++i) {
        keypoints[1].push_back(cv::Point2f(keypoints1Accessor[i][0], keypoints1Accessor[i][1]));
    }
}

void SuperGlue::MatchFeatures(bool plot) {

    // Call SuperGlue model
    torch::Dict<std::string, Tensor> input;
    input.insert("image0", image_pair_tensor[0]);
    input.insert("image1", image_pair_tensor[1]);
    input.insert("keypoints0", keypoints_tensor[0].unsqueeze(0));
    input.insert("keypoints1", keypoints_tensor[1].unsqueeze(0));
    input.insert("scores0", keypoints_scores_tensor[0].unsqueeze(0));
    input.insert("scores1", keypoints_scores_tensor[1].unsqueeze(0));
    input.insert("descriptors0", keypoints_descriptors_tensor[0].unsqueeze(0));
    input.insert("descriptors1", keypoints_descriptors_tensor[1].unsqueeze(0));

    torch::Dict<std::string, Tensor> pred;
    pred = toTensorDict(superglue.forward({input}));

    auto matches = pred.at("matches0")[0];
    auto valid = at::nonzero(matches > -1).squeeze();
    auto mkpts0 = keypoints_tensor[0].index_select(0, valid);
    auto mkpts1 = keypoints_tensor[1].index_select(0, matches.index_select(0, valid));
    auto confidence = pred.at("matching_scores0")[0].index_select(0, valid);

    auto mkpts0Accessor = mkpts0.accessor<float, 2>();
    for (int i = 0; i < mkpts0.size(0); ++i) {
        matched_points[0].push_back(cv::Point2f(mkpts0Accessor[i][0], mkpts0Accessor[i][1]));
    }

    auto mkpts1Accessor = mkpts1.accessor<float, 2>();
    for (int i = 0; i < mkpts1.size(0); ++i) {
        matched_points[1].push_back(cv::Point2f(mkpts1Accessor[i][0], mkpts1Accessor[i][1]));
    }
    
    cv::Mat imgmat0 = tensor2mat(image_pair_tensor[0]);
    imgmat0.convertTo(imgmat0, CV_8U, 255.0f);
    image_pair[0] = imgmat0;

    cv::Mat imgmat1 = tensor2mat(image_pair_tensor[1]);
    imgmat1.convertTo(imgmat1, CV_8U, 255.0f);
    image_pair[1] = imgmat1;
    
    if(plot){
        cv::Mat plot =
            make_matching_plot_fast(image_pair_tensor[0], image_pair_tensor[1], keypoints_tensor[0], keypoints_tensor[1], mkpts0, mkpts1, confidence);
        cv::imwrite("matches_superglue.png", plot);
    }
}

std::array<std::vector<cv::Point2f>, 2> SuperGlue::GetMatchedPoints() const {
    return matched_points;
}

const std::array<std::vector<cv::Point2f>, 2>& SuperGlue::GetKeypoints() const {
    return keypoints;
}

const std::array<cv::Mat, 2>& SuperGlue::GetImagePair() const {
    return image_pair;
}

std::size_t SuperGlue::GetAverageKeypointNumber() const
{
    std::size_t keypoint_number0 = keypoints[0].size();
    std::size_t keypoint_number1 = keypoints[1].size();
    return static_cast<std::size_t>((keypoint_number0 + keypoint_number1) / 2.0);
}