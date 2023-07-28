#include "experiment_designer.h"


int main()
{

    ExperimentDesigner experiment_designer;

    experiment_designer.CompareKeypointNumber();
    experiment_designer.SuperGlueRotationTranslationError();

    // for (std::size_t i = 0; i < 4; i++)
    // {
    //     experiment_designer.CompareFeatureExtractionAndBFSortTop(i);
    //     std::cout << "\n\n";
    // }

    // for (std::size_t i = 1; i < 4; i++)
    // {
    //     experiment_designer.CompareFeatureExtractionAndBFMinDistance(i);
    //     std::cout << "\n\n";
    // }

    // for (std::size_t i = 1; i < 4; i++)
    // {
    //     experiment_designer.CompareFeatureExtractionAndFLANNBased(i);
    //     std::cout << "\n\n";
    // }

    // experiment_designer.CompareCameraPoseEstimation();

    // experiment_designer.PrintMatchedImages();

    // experiment_designer.CompareDisparityMaps(0);
    // std::cout << "\n\n";
    // experiment_designer.CompareDisparityMaps(1);
    // std::cout << "\n\n";

    // experiment_designer.CompareBaselinesDisparityMaps(0);
    // std::cout << "\n\n";
    // experiment_designer.CompareBaselinesDisparityMaps(1);

    // experiment_designer.PrintDisparityMaps(5);

    return 0;

}






