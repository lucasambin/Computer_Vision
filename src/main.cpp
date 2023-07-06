/**
    This is a hand detector program based on classical computer vision techniques
    @file main.cpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <iostream>

#include "Utils.hpp"
#include "Preprocessing.hpp"
#include "Detector.hpp"
#include "Postprocessing.hpp"

#define WEIGHT_DETECTOR_NO 0.3
#define WEIGHT_DETECTOR_SKIN 1

bool checkChar(const char& a)
{
    if (a == 'y' || a == 'Y')
    {
        return true;
    }
    else if (a == 'n' || a == 'N')
    {
        return false;
    }
    return false;
}

/**
 * Main method
 */
int main(int argc, char** argv)
{
    // Set the default directory path
    std::string posImagesPath = "dataset/positive"; // position of positive images used to train the model
    std::string negImagesPath = "dataset/negative"; // position of negative images used to train the model
    std::string imagesPath = "exam/rgb"; // position of RGB images
    std::string masksPath = "exam/mask"; // position of mask images
    std::string boxesPath = "exam/det"; // position of txt file
    char trained_detector = 'y'; // test a trained detector
    std::string resDocsPath = "exam/results/doc"; // position of txt file which contains the metrics result
    std::string resBoxImagesPath = "exam/results/handBox"; // position of RGB images with Bounding box
    std::string resHandImagesPath = "exam/results/handColor"; // position of RGB images with Hand color

    // Read the directory path from console
    char choice;
    while (true)
    {        
        std::cout << "The directory path are:" << std::endl;
        std::cout << "Positive images: " << posImagesPath << std::endl;
        std::cout << "Negative images: " << negImagesPath << std::endl;
        std::cout << "RGB images: " << imagesPath << std::endl;
        std::cout << "Mask images: " << masksPath << std::endl;
        std::cout << "Bounding Box files: " << boxesPath << std::endl;
        std::cout << "Use the trained detector (y/n): " << trained_detector << std::endl;
        std::cout << "Do you want to change the position of directory? (y/n)" << std::endl;
        std::cin >> choice;
        // OK
        if (choice == 'y' || choice == 'Y')
        {
            std::cout << std::endl << "Positive images: ";
            std::cin >> posImagesPath;
            std::cout << std::endl << "Negative images: ";
            std::cin >> negImagesPath;
            std::cout << std::endl << "RGB images: ";
            std::cin >> imagesPath;
            std::cout << std::endl << "Mask images: ";
            std::cin >> masksPath;
            std::cout << std::endl << "Bounding Box files: ";
            std::cin >> boxesPath;
            std::cout << std::endl << "Use the trained detector (y/n): ";
            std::cin >> trained_detector;
            std::cout << std::endl;
        }
        // Read the directory path from console
        else if (choice == 'n' || choice == 'N')
        {
            break;
        }
        // Error
        else
        {
            std::cout << "Invalid choice! Try again" << std::endl;
        }
    }

    std::string hog_filename = "HOGHands.xml";
    std::string hog_skin_filename = "HOGHandsSkin.xml";
    std::string trained_hog_filename = "TrainedHOGHands.xml";
    std::string trained_hog_skin_filename = "TrainedHOGHandsSkin.xml";

    Preprocessing preprocessor = Preprocessing();
    Postprocessing postprocessor = Postprocessing();
    
    // Training of the model on the original images, resized with size (384,216)
    // Based on statistics, see the report for more detail:
    cv::Size patchSize = cv::Size(176, 128);
    cv::Size resImgSize = cv::Size(384, 216);
    Detector detectorNo = Detector(patchSize, resImgSize);
    Detector detectorSkin = Detector(patchSize, resImgSize, &apply_skin_threshold);

    // Create and train the model, if requested (trained_detector -> n)
    if (!checkChar(trained_detector))
    {
        std::cout << "Creating the models...\t";
        detectorNo.addTrainingSet(posImagesPath, 1);
        detectorNo.addTrainingSet(negImagesPath, -1);
        detectorNo.train();
        detectorNo.save(hog_filename);
        detectorSkin.addTrainingSet(posImagesPath, 1);
        detectorSkin.addTrainingSet(negImagesPath, -1);
        detectorSkin.train();
        detectorSkin.save(hog_skin_filename);
        std::cout << "Done" << std::endl;
        std::cout << "Saved first model as: " << hog_filename << std::endl;
        std::cout << "Saved second model as: " << hog_skin_filename << std::endl;
    }
    // Use the trained model (trained_detector -> y)
    else
    {
        // Check if the pretrained model exist
        if (detectorNo.load(trained_hog_filename))
        {
            std::cout << "Loading the pretrained model from: " << trained_hog_filename << std::endl;
        }
        // Model not found
        else
        {
            std::cout << "Pretrained model with filename: " << trained_hog_filename << " not found!" << std::endl;
            return -1;
        }
        if (detectorSkin.load(trained_hog_skin_filename))
        {
            std::cout << "Loading the pretrained skin model from: " << trained_hog_skin_filename << std::endl;
        }
        // Model not found
        else
        {
            std::cout << "Pretrained skin model with filename: " << trained_hog_skin_filename << " not found!" << std::endl;
            return -1;
        }

    }

    // Hand detection
    std::cout << "Hand Detection" << std::endl;
    // Load the test images
    std::vector<cv::Mat> testImages = preprocessor.loadImages(imagesPath);
    // Load the test masks
    std::vector<cv::Mat> testMasks = preprocessor.loadMasks(masksPath);
    // Load the ground truth from txt file
    std::vector<std::vector<cv::Rect>> testBoxes = preprocessor.loadBoxes(boxesPath);

    // Detection, since we use a specific threshold we create a copy of the test set 
    // and we use it to detect the bounding box
    std::vector<cv::Mat> testImages1;
    for (int i = 0; i < testImages.size(); i++)
    {
        testImages1.push_back(testImages.at(i).clone());
    }
    
    detectorNo.testImages(testImages1);
    detectorSkin.testImages(testImages1);
    std::vector<std::vector<cv::Rect>> detectedRectNo = detectorNo.getRects();
    std::vector<std::vector<double>> detectedScoresNo = detectorNo.getConfidenceScores();
    std::vector<std::vector<cv::Rect>> detectedRectSkin = detectorNo.getRects();
    std::vector<std::vector<double>> detectedScoresSkin = detectorNo.getConfidenceScores();
    for(auto &scores:detectedScoresNo)
    {
        std::for_each(scores.begin(), scores.end(), [](double &c){ c *= WEIGHT_DETECTOR_NO; });
    }
    for(auto &scores:detectedScoresSkin)
    {
        std::for_each(scores.begin(), scores.end(), [](double &c){ c *= WEIGHT_DETECTOR_SKIN; });
    }
    std::vector<std::vector<cv::Rect>> detectedRect;
    std::vector<std::vector<double>> detectedScores;

    detectedRect.insert(detectedRect.end(),detectedRectNo.begin(),detectedRectNo.end());
    detectedScores.insert(detectedScores.end(),detectedScoresNo.begin(),detectedScoresNo.end());
    for (int i = 0; i < detectedRect.size(); i++)
    {
        detectedRect.at(i).insert(detectedRect.at(i).end(),detectedRectSkin.at(i).begin(),detectedRectSkin.at(i).end());
        detectedScores.at(i).insert(detectedScores.at(i).end(),detectedScoresSkin.at(i).begin(),detectedScoresSkin.at(i).end());
    }
    

    // Perform non-maxima suppression and calculate the mask after it
    std::vector<std::vector<cv::Rect>> nmsResRects;
    std::vector<cv::Mat> nmsResMasks;
    for (int i = 0; i < detectedRect.size(); i++)
    {
        std::vector<cv::Rect> rect = postprocessor.nonMaxSuppression(detectedRect.at(i), detectedScores.at(i), 0.03, 0, 0);
        cv::Mat mask = mask_skin_threshold(testImages.at(i), rect);
        nmsResRects.push_back(rect);
        nmsResMasks.push_back(mask);
    }
    // Evaluate IoU and Pixel Accuracy
    std::vector<std::vector<float>> iouScores;
    std::vector<float> pixAccuracyScores;
    for (int g = 0; g < nmsResRects.size(); g++)
    {
        std::vector<float> iouScore = postprocessor.computeIOU(testBoxes.at(g), nmsResRects.at(g));
        float paScore = postprocessor.computePixelAccuracy(testMasks.at(g), nmsResMasks.at(g));
        iouScores.push_back(iouScore);
        pixAccuracyScores.push_back(paScore);
    }
    // Draw detected bounding boxes and color the hand inside it
    for (int i = 0; i < testImages.size(); i++)
    {
        cv::Mat handBoxes = testImages.at(i).clone();
        cv::Mat handColor = testImages.at(i).clone();
        // Draw detected bounding boxes and color the hand inside it 
        postprocessor.handDetection_Segmentation(nmsResMasks[i], nmsResRects[i], handBoxes, handColor);

        // Create the filename of the output file
        std::string filename;
        if (i < 9)
        {
            filename = "/0" + std::to_string(i + 1);
        }
        else
        {
            filename = "/" + std::to_string(i + 1);
        }

        // Save the metrics result in a txt file
        if (!postprocessor.writeMetrics(resDocsPath + filename + ".txt", nmsResRects.at(i), iouScores.at(i), pixAccuracyScores.at(i)))
        {
            std::cout << "Error: File " << filename << ".txt not open" << std::endl;
        }

        // Save the resulting images
        cv::imwrite(resBoxImagesPath + filename + ".jpg", handBoxes);
        cv::imwrite(resHandImagesPath + filename + ".jpg", handColor);
        // Show the resulting images
        std::cout << "Image: " << i + 1 << std::endl;
        cv::namedWindow("Bounding Box", cv::WINDOW_NORMAL);
        cv::namedWindow("Hand Color", cv::WINDOW_NORMAL);
        cv::imshow("Bounding Box", handBoxes);
        cv::imshow("Hand Color", handColor);
        cv::waitKey();
    }
    return 0;
}