#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "rotation.hpp"
#include "StateSLAMPointLandmarks.h"
#include "Plot.h"
#include "imageFeatures.h"

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration

    //std::filesystem::path cameraPath = appPath / ".." / "data" / "camera.xml";
    Camera cam;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> cam;

    // Display loaded calibration data
    cam.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int codec = cap.get(cv::CAP_PROP_FOURCC);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation

    // Initialisation

    Eigen::VectorXd mu(12); //No landmarks, only nu and eta
    Eigen::MatrixXd S(12,12); //No landmarks, only nu and eta

    mu.setZero();
    S.setZero();

    StateSLAMPointLandmarks state(Gaussian(mu, S));

    Plot plot(state, cam);

    std::vector<int> id_History;

    int temp =0;
    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        double dt = 1/fps; // or whatever eqivalent from bufer
        ArUco_Details details = detectArUco(imgin, 20); // get corners from this and pass it to event y = ...
        std::vector<std::vector<cv::Point2f>> corners = details.corners;
        std::vector<int> ids = details.id;
        //Eigen::VectorXd y = ...corners;
    


        // Process frame
        state.view() = imgin.clone(); //these die with VTK + MINGW
        // Update state

        //MeasurementPointBundle()


        // Update plot
        plot.setState(state); //these die with VTK + MINGW
        plot.render(); //these die with VTK + MINGW
        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout = plot.getFrame();
            bufferedVideoWriter.write(imgout);
        }
        //cv::waitKey(1/fps); // Adjust the wait time as needed for your frame rate
        temp++;
            // Print the ids vector
    //std::cout << "ids: ";
    for (int id : ids)
    {
        //std::cout << id << " ";

        if (std::find(id_History.begin(), id_History.end(), id) == id_History.end())
        {
            id_History.push_back(id);
        }

        
    }
    // Extract indices of elements in ids that are also in id_History
    std::vector<int> indicesInHistory;
    for (int id : ids)
    {
        auto it = std::find(id_History.begin(), id_History.end(), id);
        if (it != id_History.end())
        {
            // Calculate the index and push it to the indicesInHistory vector
            int index = std::distance(id_History.begin(), it);
            indicesInHistory.push_back(index);
        }
    }

    // Print the indicesInHistory vector
    std::cout << "Indices in History: ";
    for (int index : indicesInHistory)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;


    }



    bufferedVideoReader.stop();
    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
}
