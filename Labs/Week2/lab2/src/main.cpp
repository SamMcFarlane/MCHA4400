#include <cstdlib>
#include <iostream>
#include <string>  
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | orb      | feature detector to use (e.g., harris, shi, aruco, orb)}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    cv::String detector = parser.get<std::string>("detector");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::cout << "File: " << inputPath.string() << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (doExport)
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

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::cout << "Output name: " << outputPath.string() << std::endl;
    }

    // Check if input is an image or video (or neither)
    //bool isVideo = false; // TODO
    //bool isImage = true; // TODO

    cv::VideoCapture cap(inputPath.string());
    bool isVideo = (cap.get(cv::CAP_PROP_FRAME_COUNT)>1);
    bool isImage = !isVideo;

    if (!isImage && !isVideo)
    {
        std::cout << "Could not read file: " << inputPath.string() << std::endl;
        return EXIT_FAILURE;
    }

    if (isImage)
    {
        // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        cv::Mat image = cv::imread(inputPath.string());
        cv::Mat imageDraw;
        if(detector=="harris")
            imageDraw = detectAndDrawHarris(image, maxNumFeatures);
        else if(detector=="shi")
            imageDraw = detectAndDrawShiAndTomasi(image, maxNumFeatures);
        else if(detector=="orb")
            imageDraw = detectAndDrawORB(image, maxNumFeatures);
        else if(detector=="aruco")
            imageDraw = detectAndDrawArUco(image, maxNumFeatures);
        else
            std::cout<<"Error in detector - COOKED" << std::endl;
        
        if (doExport)
        {
            // TODO: Write image returned from detectAndDraw to outputPath
            cv::imwrite(outputPath.string(),imageDraw);
        }
        else
        {
            // TODO: Display image returned from detectAndDraw on screen and wait for keypress
            cv::imshow("Image",imageDraw);
            cv::waitKey(0);
        }
    }

    if (isVideo)
    {
        //cv::VideoCapture cap(inputPath.string());

        if (!cap.isOpened())
        {
            std::cerr << "Error: Cannot open video file." << std::endl;
            return EXIT_FAILURE;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

        cv::VideoWriter writer;

        if (doExport)
        {
            // TODO: Open output video for writing using the same fps as the input video
            //       and the codec set to cv::VideoWriter::fourcc('m', 'p', '4', 'v')
             writer.open("../out/output_video.mp4", fourcc, fps, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
            if (!writer.isOpened())
            {
                std::cerr << "Error: Cannot open output video file." << std::endl;
                return EXIT_FAILURE;
            }


        }

        while (true)
        {
            // TODO: Get next frame from input video

            cv::Mat frame;
            cv::Mat imageDraw;
            cap>>frame;

            // TODO: If frame is empty, break out of the while loop
            if(frame.empty()){
                std::cout<<"Frame is empty"<<std::endl;
                break;
            }
            
            // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line

            if(detector=="harris")
                imageDraw = detectAndDrawHarris(frame, maxNumFeatures);
            else if(detector=="shi")
                imageDraw = detectAndDrawShiAndTomasi(frame, maxNumFeatures);
            else if(detector=="orb")
                imageDraw = detectAndDrawORB(frame, maxNumFeatures);
            else if(detector=="aruco")
                imageDraw = detectAndDrawArUco(frame, maxNumFeatures);
            else
                std::cout<<"Error in detector - COOKED" << std::endl;

            if (doExport)
            {
                // TODO: Write image returned from detectAndDraw to frame of output video
                writer.write(imageDraw);
            }
            else
            {
                // TODO: Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Processed Frame", imageDraw);
                cv::waitKey(1000 / fps);
            }
        }

        // TODO: release the input video object
        cap.release();

        if (doExport)
        {
            // TODO: release the output video object
            writer.release();
        }
    }

    return EXIT_SUCCESS;
}



