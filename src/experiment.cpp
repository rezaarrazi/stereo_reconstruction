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

    experiment_designer.PrintDisparityMaps(0);

    // experiment_designer.CompareDisparityMaps(1, 0, 0);
    // std::cout << "\n\n";
    // experiment_designer.CompareDisparityMaps(1, 0, 1);
    // std::cout << "\n\n";
    // experiment_designer.CompareDisparityMaps(3, 2, 0);
    // std::cout << "\n\n";
    // experiment_designer.CompareDisparityMaps(3, 2, 1);
    // std::cout << "\n\n";

    return 0;

}






