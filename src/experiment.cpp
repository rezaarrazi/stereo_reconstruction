#include "experiment_designer.h"


int main()
{

    ExperimentDesigner experiment_designer;

    // experiment_designer.CompareKeypointNumber();

    // for (std::size_t i = 0; i < 3; i++)
    // {
    //     experiment_designer.CompareFeatureExtractionAndMatching(i);
    //     std::cout << "\n\n";
    // }

    // experiment_designer.CompareCameraPoseEstimation();

    experiment_designer.CompareDisparityMaps();

    return 0;

}






