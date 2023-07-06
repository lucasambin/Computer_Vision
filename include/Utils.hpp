/**
    Declaration of a class that provides some utility methods
    @file Utils.hpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/

#ifndef UTILS_H
#define UTILS_H 
#include <opencv2/opencv.hpp>

const int SKIN_TYPE_1 = 0;
const int SKIN_TYPE_2 = 1;

bool pixel_color_check(uchar r, uchar g, uchar b, uchar cr, uchar cb, uchar h);



// Compute mask
cv::Mat mask_color_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes, int type);
void mask_image(cv::Mat& src, cv::Mat mask);
cv::Mat apply_skin_threshold(cv::Mat& src);
// Compute mask given a box
cv::Mat mask_skin_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes);
cv::Mat mask_skin_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes, int type);
#endif