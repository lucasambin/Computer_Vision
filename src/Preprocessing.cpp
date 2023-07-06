/**
    Definition of a class that performs all the preprocessing operations needed before feature extraction and Detection parts
    @file Preprocessing.cpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#include "Preprocessing.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>


/**
    Constructor
*/
Preprocessing::Preprocessing() 
{}

/**
    Load images from the specified directory
    @param dirname Path from where to load the images
    @return A vector of images
*/
std::vector<cv::Mat> Preprocessing::loadImages(const std::string& dirname)
{
    // We assume that the folder is correct (exam/rgb)
    std::vector<std::string> imagesName;
    cv::glob(dirname + "/*.jpg", imagesName, 0);
    std::vector<cv::Mat> images;
    // Loads all the images of the folder (RGB image)
    for (int i = 0; i < imagesName.size(); i++)
    {
        cv::Mat aImage = cv::imread(imagesName.at(i));
        if (aImage.empty())
        {
            std::cout << imagesName.at(i) << " is invalid!" << std::endl;
            continue;
        }
        images.push_back(aImage);
    }
    return images;
}


/**
    Load masks from the specified directory
    @param dirname Path from where to load the masks
    @return A Vector of images
*/
std::vector<cv::Mat> Preprocessing::loadMasks(const std::string& dirname)
{
    // We assume that the folder is correct (exam/mask)
    std::vector<std::string> masksName;
    cv::glob(dirname + "/*.png", masksName, 0);
    std::vector<cv::Mat> masks;
    // Loads all the masks of the folder (Grayscale image)
    for (int i = 0; i < masksName.size(); i++)
    {
        cv::Mat aMask = cv::imread(masksName.at(i), cv::IMREAD_GRAYSCALE);
        if (aMask.empty())
        {
            std::cout << masksName.at(i) << " is invalid!" << std::endl;
            continue;
        }
        masks.push_back(aMask);
    }
    return masks;
}


/**
    Load bounging boxes from the specified directory
    @param dirname Path from where to load the bounding boxes
    @return A vector for each document of bounding boxes
*/
std::vector<std::vector<cv::Rect>> Preprocessing::loadBoxes(const std::string& dirname)
{
    // We assume that the folder is correct (exam/det)
    std::vector<std::string> boxesName;
    cv::glob(dirname + "/*.txt", boxesName, 0);
    std::vector<std::vector<cv::Rect>> boxes;
    // Loads all the bounding boxes of the folder
    for (int i = 0; i < boxesName.size(); i++)
    {
        // Load all the bounding box of a document
        std::vector<cv::Rect> aBox;
        int x = 0, y = 0, w = 0, h = 0;
        std::ifstream aDoc;
        aDoc.open(boxesName.at(i), std::ios_base::in);
        while (aDoc >> x >> y >> w >> h)
        {
            aBox.push_back(cv::Rect(x, y, w, h));
        }
        aDoc.close();
        // Load all the previous bounding box into a new vector
        boxes.push_back(aBox);
    }
    return boxes;
}