/**
    Declaration of a class that performs all the preprocessing operations needed before feature extraction and Detection parts
    @file Preprocessing.hpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <opencv2/opencv.hpp>

class Preprocessing
{
    // Methods

    public:
        // Constructor
        Preprocessing();
    
        // Function to load images
        std::vector<cv::Mat> loadImages(const std::string& dirname);
        // Function to load masks
        std::vector<cv::Mat> loadMasks(const std::string& dirname);
        // Function to load bounding boxes
        std::vector<std::vector<cv::Rect>> loadBoxes(const std::string& dirname);
};
#endif