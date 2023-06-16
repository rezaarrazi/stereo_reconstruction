#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>

typedef unsigned char BYTE;

class VirtualSensor
{
public:

	VirtualSensor() 
	{

	}

	~VirtualSensor()
	{
		
	}

	bool Init(const std::string& datasetDir)
	{
		m_baseDir = datasetDir;

        // Load images and disparities using OpenCV
        m_image0 = cv::imread(m_baseDir + "im0.png", cv::IMREAD_COLOR);
        m_image1 = cv::imread(m_baseDir + "im1.png", cv::IMREAD_COLOR);
        LoadPFMFile(m_baseDir + "disp0.pfm", m_disparity0);
        LoadPFMFile(m_baseDir + "disp1.pfm", m_disparity1);

        // Load calibration information
        LoadCalibFile(m_baseDir + "calib.txt");

        return true;
	}

	cv::Mat GetDisparity0()
    {
        return m_disparity0;
    }

    cv::Mat GetDisparity1()
    {
        return m_disparity1;
    }

    cv::Mat GetImage0()
    {
        return m_image0;
    }

    cv::Mat GetImage1()
    {
        return m_image1;
    }

	Eigen::Matrix3f GetIntrinsics0()
    {
        return m_cam0Intrinsics;
    }

    Eigen::Matrix3f GetIntrinsics1()
    {
        return m_cam1Intrinsics;
    }

private:

	// Add function to load PFM file
    void LoadPFMFile(const std::string& filename, cv::Mat& dst)
    {
        // Read PFM file
        dst = cv::imread(filename, cv::IMREAD_UNCHANGED);

        // Handle infinite values
        dst.setTo(0, dst == std::numeric_limits<float>::infinity());

        // Normalize and convert to uint8
        cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    // Add function to load calibration file
    void LoadCalibFile(const std::string& filename)
	{
		// Open the file
		std::ifstream calibFile(filename);
		if (!calibFile.is_open())
		{
			std::cerr << "Could not open the calibration file: " << filename << std::endl;
			return;
		}

		// Parse the file
		Eigen::Matrix3f cam0, cam1;
		float doffs, baseline;
		unsigned int width, height, ndisp, isint, vmin, vmax;

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				calibFile >> cam0(i, j);
			}
		}

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				calibFile >> cam1(i, j);
			}
		}

		calibFile >> doffs;
		calibFile >> baseline;
		calibFile >> width;
		calibFile >> height;
		calibFile >> ndisp;
		calibFile >> isint;
		calibFile >> vmin;
		calibFile >> vmax;

		m_cam0Intrinsics = cam0;
		m_cam1Intrinsics = cam1;
		m_doffs = doffs;
		m_baseline = baseline;

		// Close the file
		calibFile.close();
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	std::string m_baseDir;

	cv::Mat m_disparity0;
    cv::Mat m_disparity1;
    cv::Mat m_image0;
    cv::Mat m_image1;

	Eigen::Matrix3f m_cam0Intrinsics;
    Eigen::Matrix3f m_cam1Intrinsics;
    float m_doffs;
    float m_baseline;
};
