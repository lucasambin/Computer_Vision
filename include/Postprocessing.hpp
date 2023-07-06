/**
    Declaration of a class that executes the postprocessing operations
    @file Postprocessing.hpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H
#include <opencv2/opencv.hpp>

class Postprocessing
{
    // Methods

    public:
        // Constructor
        Postprocessing();
    
        // Perform non-maxima suppression
        std::vector<cv::Rect> nonMaxSuppression(const std::vector<cv::Rect>& srcRects, const std::vector<double>& scores, float thresh, int neighbors, double minScoresSum);
        // Produce the output images (hand detection & hand segmentation)
        void handDetection_Segmentation(const cv::Mat& mask, const std::vector<cv::Rect>& bBoxes, cv::Mat& handBoxes, cv::Mat& handColor);
        // Test the performances of the detector by using IoU metric
        std::vector<float> computeIOU(const std::vector<cv::Rect>& groundTruth, const std::vector<cv::Rect>& detections);
        // Compute Pixel Accuracy between two mask
        float computePixelAccuracy(const cv::Mat& examMask, const cv::Mat detMask);
        // Write metrics result
        bool writeMetrics(const std::string& filename, const std::vector<cv::Rect>& resBoxes, const std::vector<float>& iouScores, const float& pixAccuracyScores);

    private:
        // Compute Intersection over Union between boxes (IoU)
        float computeOneIOU(const cv::Rect& box1, const cv::Rect& box2);
};

#endif