#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"
#include "camera_pose_estimator.h"
#include "dense_matcher.h"
#include "scene_reconstructor.h"
#include "superglue.h"


int main()
{
    int index = 0;
    StereoDataset stereo_dataset;
    DenseMatcher dense_matcher;
    SceneReconstructor scene_reconstructor;

    std::array<std::size_t, 2> image_indices = {0, 5};
    std::array<std::string, 2> file_indices = {"1", "2"};
    std::array<std::string, 2> dense_matcher_names = {"StereoBM", "StereoSGBM"};
    float distance_threshold = 20000.0;

    for (std::size_t i = 0; i < 2; i++)
    {
        stereo_dataset.SetImages(image_indices[i]);

        stereo_dataset.SetCalibrations(image_indices[i]);

        stereo_dataset.SetDisparityMaps(image_indices[i]);

        dense_matcher.LoadDataDirectly(stereo_dataset);

        for (std::size_t j = 0; j < 2; j++)
        {
            dense_matcher.ComputeDisparityMapDirectly(j);
            scene_reconstructor.LoadData(stereo_dataset);
            scene_reconstructor.ReconstructScene(dense_matcher.GetDisparityMap(), distance_threshold);
            scene_reconstructor.WriteMeshToFile("../" + file_indices[i] + dense_matcher_names[j] + "mesh.off");
        }

        scene_reconstructor.ReconstructScene(stereo_dataset.GetDisparityMaps()[0], distance_threshold);

        scene_reconstructor.WriteMeshToFile("../" + file_indices[i] + "gtmesh.off");
    }



    // StereoDataset stereo_dataset;
    // stereo_dataset.SetImages(index);

    // FeatureExtractor feature_extractor;
    // feature_extractor.SetImages(stereo_dataset.GetImages());

    // feature_extractor.ExtractFeatures(0);

    // SparseMatcher sparse_matcher;

    // sparse_matcher.MatchSparselyBFSortTop(feature_extractor.GetKeypoints(), feature_extractor.GetFeatures(), 50);
    
    // SuperGlue superglue("./superglue/SuperPoint.zip", "./superglue/SuperGlue.zip");
    // superglue.SetImages(stereo_dataset.GetImagePairPath()[0], stereo_dataset.GetImagePairPath()[1], 640);
    // superglue.ExtractFeatures();
    // superglue.MatchFeatures(true);

    // // sparse_matcher.DisplayMatchings(stereo_dataset.GetImages(), feature_extractor.GetKeypoints(), true);

    // stereo_dataset.SetCalibrations(0);

    // CameraPoseEstimator camera_pose_estimator;
    // camera_pose_estimator.SetCameraIntrinsics(stereo_dataset.GetCameraIntrinsics());
    // // camera_pose_estimator.SetMatchedPoints(sparse_matcher.GetMatchedPoints());
    // camera_pose_estimator.SetMatchedPoints(superglue.GetMatchedPoints());

    // camera_pose_estimator.EstimateCameraPose(3);

    // cv::Mat rotation = camera_pose_estimator.GetRotation();
    // cv::Mat translation = camera_pose_estimator.GetTranslation();

    // std::cout << rotation << '\n';
    // std::cout << translation << '\n';

    // DenseMatcher dense_matcher;
    // dense_matcher.LoadData(stereo_dataset, rotation, translation);
    // dense_matcher.RectifyImages();
    // dense_matcher.ComputeDisparityMap(1);
    // dense_matcher.LoadDataDirectly(stereo_dataset);
    // dense_matcher.ComputeDisparityMapDirectly(1);

    // stereo_dataset.SetDisparityMaps(index);
    
    // float distance_threshold = 20000.0;
    // SceneReconstructor scene_reconstructor;
    // scene_reconstructor.LoadData(stereo_dataset);
    // // scene_reconstructor.ReconstructScene(dense_matcher.GetDisparityMap(), distance_threshold);
    // scene_reconstructor.ReconstructScene(stereo_dataset.GetDisparityMaps()[0], distance_threshold);
    // scene_reconstructor.WriteMeshToFile("../mesh.off");

    // cv::Mat out1;
    // cv::Mat out2;

    // // cv::resize(dense_matcher.GetColorfulDisparityMap(), out1, cv::Size(), 0.5, 0.5);
    // // cv::imshow("disparity", out1);

    // cv::applyColorMap(stereo_dataset.GetDisparityMaps()[0], out2, cv::COLORMAP_JET);
    // cv::resize(out2, out2, cv::Size(), 0.5, 0.5);
    // cv::imshow("disparity_gt", out2);

    // cv::waitKey(0);

    return 0;

}








