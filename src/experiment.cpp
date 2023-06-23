#include "experiment_designer.h"


int main()
{

    ExperimentDesigner experiment_designer;

    // experiment_designer.CompareKeypointNumber();

    for (std::size_t i = 0; i < 3; i++)
    {
        experiment_designer.CompareCameraPoseEstimation(i);
        std::cout << "\n\n";
    }

    return 0;

}






