/**
    This class performs all the postprocessing operations needed after the detection part
    @file Postprocessing.cpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/

#include "Postprocessing.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <assert.h>

/**
 * Constructor
 */
Postprocessing::Postprocessing() 
{}

/**
 * Compute non-maxima suppression to remove overlapping bounding boxes
 */
std::vector<cv::Rect> Postprocessing::nonMaxSuppression(const std::vector<cv::Rect>& srcRects, const std::vector<double>& scores, float thresh, int neighbors = 0, double minScoresSum = 0)
{
    std::vector<cv::Rect> resRects;
    const size_t size = srcRects.size();
    // srcRects are empty, return an empty vector
    if (size == 0)
    {
        // No bounding box detected
        return resRects;
    }
    assert(srcRects.size() == scores.size());
    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(scores[i], i);
    }
    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];
        int neigborsCount = 0;
        float scoresSum = lastElem->first;
        idxs.erase(lastElem);
        for (auto pos = std::begin(idxs); pos != std::end(idxs);)
        {
            // grab the current rectangle..
            const cv::Rect& rect2 = srcRects[pos->second];
            float overlap = computeOneIOU(rect1, rect2);
            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum)
        {
            resRects.push_back(rect1);
        }
    }
    return resRects;
}

/**
 * Compute non-maxima suppression to remove overlapping bounding boxes
 */
void Postprocessing::handDetection_Segmentation(const cv::Mat& mask, const std::vector<cv::Rect>& bBoxes, cv::Mat& handBoxes, cv::Mat& handColor)
{
    // Random generator number
    cv::RNG rng(12345);

    // Assume that bBox and handColor are just the clone of the source image
    for (size_t i = 0; i < bBoxes.size(); i++)
    {
        int b = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int r = rng.uniform(0, 256);
        cv::Scalar color = cv::Scalar(b, g, r);
        cv::rectangle(handBoxes, bBoxes[i].tl(), bBoxes[i].br(), color, 2);
        int xoff = bBoxes[i].x;
        int yoff = bBoxes[i].y;
        cv::Mat roi = mask(bBoxes[i]);
        // Color original image
        for (int i = 0; i < roi.rows; i++)
        {
            for (int j = 0; j < roi.cols; j++)
            {
                // If white pixel in the mask image => color output image
                if (roi.at<unsigned char>(i, j) == 255)
                {
                    handColor.at<cv::Vec3b>(yoff + i, xoff + j)[0] = b;
                    handColor.at<cv::Vec3b>(yoff + i, xoff + j)[1] = g;
                    handColor.at<cv::Vec3b>(yoff + i, xoff + j)[2] = r;
                    // addWeighted for transparecy
                }
            }
        }
    }
}


/**
 * Compute Pixel Accuracy between two masks
 * @param examMask is the mask loaded from the folder
 * @param detMask is the resulting mask detected by us
 * @return The Pixel Accuracy value
 */
float Postprocessing::computePixelAccuracy(const cv::Mat& examMask, const cv::Mat detMask)
{
    int tp = 0; // True Positive pixel classified correctly as Ground truth
    int fp = 0; // False Positive pixel classified incorrectly as Ground truth
    int tn = 0; // True Negative pixel classified correctly as not Ground truth
    int fn = 0; // False Negative pixel classified incorrectly as not Ground truth

    // We assume that the two images have the same size
    for (int i = 0; i < examMask.rows; i++)
    {
        for (int j = 0; j < examMask.cols; j++)
        {
            unsigned char gtPix = examMask.at<unsigned char>(i, j);
            unsigned char detPix = detMask.at<unsigned char>(i, j);
            // The gtPixel is white
            if (gtPix == 255)
            {
                // If detPixel is white = > pixel is correctly classified
                if (detPix == 255)
                {
                    tp++;
                }
                // If detPixel is black => pixel is incorrectly classified
                else
                {
                    fn++;
                }
            }
            // The gtPixel is black
            else
            {
                // If detPixel is white = > pixel is incorrectly classified
                if (detPix == 255)
                {
                    fp++;
                }
                // If detPixel is black => pixel is correctly classified
                else
                {
                    tn++;
                }
            }
        }
    }
    // Compute and return Pixel Accuracy
    return float(tp + tn) / float(tp + tn + fp + fn);
}


/**
 * Test the performances of the detector by using IoU metric
 * @param groundTruth The vector of the real bounding boxes
 * @param detections The vector of the detected bounding boxes
 * @returns IoU scores
 */
std::vector<float> Postprocessing::computeIOU(const std::vector<cv::Rect>& groundTruth, const std::vector<cv::Rect>& detections)
{
    // max obtained iou for each detected box.
    std::vector<float> iouScores;
    for (int i = 0; i < detections.size(); i++)
    {
        std::vector<float> tmp;
        for (int j = 0; j < groundTruth.size(); j++)
        {            
            float iou = computeOneIOU(detections.at(i), groundTruth.at(j));
            tmp.push_back(iou);
        }
        float maxElem = -1.f;
        for (int i = 0; i < tmp.size(); i++)
        {
            if (tmp.at(i) > maxElem)
                maxElem = tmp.at(i);
        }
        iouScores.push_back(maxElem);
    }
    return iouScores;
}

bool Postprocessing::writeMetrics(const std::string& filename, const std::vector<cv::Rect>& resBoxes, const std::vector<float>& iouScores, const float& pixAccuracyScores)
{
    std::ofstream file(filename, std::ofstream::out);
    if (file.is_open())
    {
        file << "Pixel Accuracy score: " << pixAccuracyScores << std::endl;
        file << "Number of Bounding Box detected: " << resBoxes.size() << std::endl;
        for (int i = 0; i < resBoxes.size(); i++) // Check if the file is successfully opened for writing
        {            
            cv::Rect aResBox = resBoxes.at(i);
            float aIouScore = iouScores.at(i);
            file << "Bounding Box: " << i + 1 << std::endl;
            file << "Result Bounding Box: " << aResBox.x << " " << aResBox.y << " " << aResBox.width << " " << aResBox.height << std::endl;
            file << "IoU score: " << aIouScore << std::endl << std::endl;
        }
    }
    else
    {        
        return false;
    }
    file.close();
    return true;
}

// Private Methods

/**
 * Compute IoU between two bounding boxes
 * @param box1 One of the two boxes
 * @param box2 One of the two boxes
 * @return The IoU value
 */
float Postprocessing::computeOneIOU(const cv::Rect& box1, const cv::Rect& box2)
{
    // determine the(x, y) - coordinates of the intersection rectangle
    int xA = std::max(box1.x, box2.x);
    int yA = std::max(box1.y, box2.y);
    int xB = std::min(box1.x + box1.width, box2.x + box2.width);
    int yB = std::min(box1.y + box1.height, box2.y + box2.height);


    // compute the area of intersection rectangle
    float interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);

    // compute the area of both the predictionand ground - truth rectangles
    float boxAArea = (box1.width + 1) * (box1.height + 1);
    float boxBArea = (box2.width + 1) * (box2.height + 1);

    // compute the intersection over union by taking the intersection
    // areaand dividing it by the sum of prediction + ground - truth
    // areas - the interesection area
    float iou = interArea / (boxAArea + boxBArea - interArea);
    // return the intersection over union value
    return iou;
}

