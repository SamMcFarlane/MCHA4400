#include <cstdlib>
#include <iostream>
#include <filesystem>
#include "Camera.h"
#include "confidenceRegionDemo.h"

int main(int argc, char* argv[])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |        | print this message}"
        "{calibrate c     |        | perform camera calibration}"
        "{export e        |        | export files to the ./out/ directory}"
        "{interactive i   | 2      | interactivity (0:none, 1:last image, 2:all images)}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 7");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool hasCalibrate = parser.has("calibrate");
    bool hasExport = parser.has("export");
    int interactive = parser.get<int>("interactive");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (hasExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::cout << "Creating directory " << outputDirectory.string() << std::endl;
            std::filesystem::create_directory(outputDirectory);
        }
        std::cout << "Output directory set to " << outputDirectory.string() << std::endl;
    }

    // Get calibration configuration
    std::filesystem::path appPath = parser.getPathToApplication();
    std::filesystem::path configPath = appPath / ".." / "data" / "config.xml";

    if (!std::filesystem::exists(configPath))
    {
        std::cout << "File: " << configPath << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Camera calibration file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";

    Camera cam;
    if (hasCalibrate)
    {
        // Calibrate camera from chessboard data
        cam.calibrate(chessboardData);

        // Write camera calibration to file
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
        fs << "camera" << cam;
        fs.release();
    }
    else
    {
        // Do confidence region demo

        // Read camera calibration using default camera file path
        if (!std::filesystem::exists(cameraPath))
        {
            std::cout << "File: " << cameraPath << " does not exist" << std::endl;
            return EXIT_FAILURE;
        }
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        assert(fs.isOpened());
        fs["camera"] >> cam;

        // Run calibration confidence region demo
        calibrationConfidenceRegionDemo(cam, chessboardData, outputDirectory, interactive);
    }

    return EXIT_SUCCESS;
}
