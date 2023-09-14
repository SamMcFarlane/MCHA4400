#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <filesystem>
#include "Camera.h"
#include "serialisation.hpp" // Include the serialization templates
#include "calibrate.h"
#include "BufferedVideo.h"   // Include the BufferedVideo classes

void calibrateCamera(const std::filesystem::path & configPath)
{
    // TODO
    // - Read XML at configPath
    // - Parse XML and extract relevant frames from source video containing the chessboard
    // - Perform camera calibration
    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // - Visualise the camera calibration results


     // Read XML at configPath
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Unable to open configuration file." << std::endl;
        return;
    }

    // Read chessboard data from the configuration file
    ChessboardData chessboardData(configPath);

    std::cout<<"I'm getting here after chessboardData"<<std::endl;

    std::string videoFilePath = "../data/calibration.MOV";
    // Open the video file
    cv::VideoCapture videoCapture(videoFilePath);
    if (!videoCapture.isOpened())
    {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return;
    }
    std::cout<<"I'm getting here after videoCapture"<<std::endl;

    // Visualize the chessboard corners (optional)
    cv::namedWindow("Chessboard Corners", cv::WINDOW_NORMAL);

    // Loop over frames in the video
    cv::Mat frame;
    int frameCounter = 0;
    int frameSamples = 30; // Specifies every number of frames to read (e.g., if 10, reads every 10th frame)
    int maxNumImages = 30; // max number of images to calibrate off
    int maxNumImagesCounter = 0;
    while (videoCapture.read(frame)&&(maxNumImagesCounter!=maxNumImages))
    {
        // Check if the frame is empty (end of video)
        if (frame.empty())
        {
            break;  // Exit the loop if there are no more frames
        }
        frameCounter++;

        if (frameCounter % frameSamples != 0) {
            continue;  // Skip frames until the 10th frame
        }
        // Parse XML and extract relevant frames from source video containing the chessboard
        ChessboardImage ci(frame, chessboardData.chessboard, "");
        if(ci.isFound){
            maxNumImagesCounter++;
            chessboardData.chessboardImages.push_back(ci);

        // Visualize the chessboard corners (optional)
            ci.drawCorners(chessboardData.chessboard);
            cv::imshow("Chessboard Corners", ci.image);
        }
        int key = cv::waitKey(10);  // Wait for 10 milliseconds and capture keyboard input
        if (key == 27) {  // 27 is the ASCII value for the Escape key
            // Escape key pressed, exit the loop
            break;
        }
    }


    // Close the video file
    videoCapture.release();
    cv::destroyAllWindows();

    // Perform camera calibration
    Camera camera;
    camera.calibrate(chessboardData);

    // Write the camera matrix and lens distortion parameters to camera.xml file
    cv::FileStorage outputFs("../data/camera.xml", cv::FileStorage::WRITE);
    //outputFs << "CameraMatrix" << camera.cameraMatrix;
    //outputFs << "DistortionCoefficients" << camera.distCoeffs;
    outputFs<<"camera"<<camera; 
    outputFs.release();

}