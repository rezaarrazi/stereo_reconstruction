#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#include "Eigen.h"
#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

struct Triangle {
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;

	Triangle() : idx0{ 0 }, idx1{ 0 }, idx2{ 0 } {}

	Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
		idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = width*height;

	// TODO: Determine number of valid faces
	std::vector<Triangle> triangles;
	for (unsigned int i = 0; i < height - 1; i++) {
		for (unsigned int j = 0; j < width - 1; j++) {
			unsigned int i0 = i*width + j;
			unsigned int i1 = (i + 1)*width + j;
			unsigned int i2 = i*width + j + 1;
			unsigned int i3 = (i + 1)*width + j + 1;

			bool valid0 = vertices[i0].position.allFinite();
			bool valid1 = vertices[i1].position.allFinite();
			bool valid2 = vertices[i2].position.allFinite();
			bool valid3 = vertices[i3].position.allFinite();

			if (valid0 && valid1 && valid2) {
				float d0 = (vertices[i0].position - vertices[i1].position).norm();
				float d1 = (vertices[i0].position - vertices[i2].position).norm();
				float d2 = (vertices[i1].position - vertices[i2].position).norm();
				if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2){
					Triangle triangle(i0, i1, i2);
					triangles.push_back(triangle);
				}
			}
			if (valid1 && valid2 && valid3) {
				float d0 = (vertices[i3].position - vertices[i1].position).norm();
				float d1 = (vertices[i3].position - vertices[i2].position).norm();
				float d2 = (vertices[i1].position - vertices[i2].position).norm();
				if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2){
					Triangle triangle(i1, i3, i2);
					triangles.push_back(triangle);
				}
			}
		}
	}
	unsigned nFaces = triangles.size();

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;

	outFile << "# numVertices numFaces numEdges" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	for (unsigned int i = 0; i < nVertices; ++i)
	{
		const auto& vertex = vertices[i];
		if (vertex.position.allFinite())
			outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z() << " "
			<< int(vertex.color.x()) << " " << int(vertex.color.y()) << " " << int(vertex.color.z()) << " " << int(vertex.color.w()) << std::endl;
		else
			outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
	}

	// TODO: save valid faces
	outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

	for (unsigned int i = 0; i < triangles.size(); i++) {
		outFile << "3 " << triangles[i].idx0 << " " << triangles[i].idx1 << " " << triangles[i].idx2 << std::endl;
	}
	
	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	std::string filenameIn = "../Data/chess1/";
	std::string filenameBaseOut = "mesh_";

	// load dataset
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

    cv::imshow("image0", sensor.GetImage0());
    int k = cv::waitKey(0); // Wait for a keystroke in the window

	return 0;
}