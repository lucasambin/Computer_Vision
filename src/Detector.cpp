/**
    This class performs the detection of objects based on the previously extracted features
    @file Detection.cpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#include "Detector.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <filesystem>

/**
    Constructor
    @param box_size the window size to be used by the Hog Descriptor
    @param resize_image_size the size of the resulting image
    @param preprocessing the function to be applied on the images before training and testing is carried out.
*/
Detector::Detector(cv::Size box_size, cv::Size resize_image_size,cv::Mat (*preprocessing)(cv::Mat&))
{
    res_img_size = resize_image_size;
    winSize = box_size;
    blockSize = cv::Size(16,16);
    blockStride = cv::Size(8,8);
    cellSize = cv::Size(8,8);
    nbins = 9;
    derivAperture = 1;
    descriptor = cv::HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture);
    //
    preprocess = preprocessing;
    //
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::EPS_SVR);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,1000,1e-3));
    svm->setGamma(0);
    svm->setNu(0.5);
    svm->setP(0.1);
    svm->setC(0.01);
    //
}
/**
 * Reads all images names from the directory provided in input.
 * @param dirname name of the parent directory from which to obtain the images names.
 * @return collection of name of files inside sush directory
*/
std::vector<std::string> Detector::readTrainingFiles(std::string dirname)
{
    std::vector<std::string> ret;
    for (const auto & entry : std::filesystem::directory_iterator(dirname))
    {
        ret.push_back(entry.path().string());
    }
    return ret;
}
/**
 * Loads a set of images from a directory and associates them with a corresponding label.
 * @param dirname name of the parent directory from which to load the images
 * @param label label to be associated with the set of data, provided as a vector of the same length of the images.
*/
void Detector::addTrainingSet(std::string dirname, std::vector<int> labels)
{
    std::vector<std::string> filenames = readTrainingFiles(dirname);
    for(int i = 0; i< filenames.size();i++)
    {
        accumulator.push_back(HogTemplate(filenames[i],labels[i]));
    }
}
/**
 * Loads a set of images from a directory and associates them with a corresponding label.
 * @param dirname name of the parent directory from which to load the images
 * @param label label to be associated with the set of data.
*/
void Detector::addTrainingSet(std::string dirname, int label)
{
    std::vector<std::string> filenames = readTrainingFiles(dirname);
    for(int i = 0; i< filenames.size();i++)
    {
        accumulator.push_back(HogTemplate(filenames[i],label));
    }
}
/**
 * Loads a dataset on the Detector, in particular a series of labels as folders is expected to be inside the directory provided in input.
 * @param dirname name of the parent directory from which to load the dataset
*/
void Detector::addTrainingSet(std::string dirname)
{
    for (const auto & entry : std::filesystem::directory_iterator(dirname))
    {
        if (entry.is_directory())
        {
            char *p;
            int converted = strtol(entry.path().string().substr(entry.path().string().find_last_of("/\\") + 1).c_str(), &p,10);
            if(!(*p))
            {
                this->addTrainingSet(entry.path().string(),converted);
            }
        }
    }
}
/**
 * Computes the Histogram of Oriented Gradients on the dataset added to the Detector.
*/
void Detector::computeImagesFeatures()
{
    for(HogTemplate& entry: accumulator)
    {
        cv::Mat img = cv::imread(entry.filename);
        // Resize Image
        cv::resize(img, img, res_img_size);
        if(preprocess!=nullptr)
        {
            preprocess(img);
        }
        /*cv::Mat resized;
        cv::resize(img,resized,winSize,0.0,0.0,cv::INTER_AREA);
        fastNlMeansDenoisingColored(resized, resized);
        std::vector<float> descs;
        descriptor.compute(resized,descs);
        entry.descriptors.insert(entry.descriptors.end(), descs.begin(), descs.end());*/
        //fastNlMeansDenoisingColored(img, img);
        std::vector<float> descs;
        cv::Mat gray;
        if (img.cols >= winSize.width && img.rows >= winSize.height)
        {
            cv::Rect r = cv::Rect((img.cols - winSize.width) / 2,
                          (img.rows - winSize.height) / 2,
                          winSize.width,
                          winSize.height);
            cvtColor(img(r), gray, cv::COLOR_BGR2GRAY);
            descriptor.compute(gray,descs);
            entry.descriptors.insert(entry.descriptors.end(), descs.begin(), descs.end());
        }
    }
}

/**
 * Returns the Histogram of Oriented Gradients for a single image.
 * @param img Image for which the histogram needs to be computed.
 * @return values of the obtained histogram.
*/
std::vector<float> Detector::getHogForImage(cv::Mat img)
{
    std::vector<float> ret;
    cv::Mat resized;
    cv::resize(img,resized, winSize,0.0,0.0,cv::INTER_AREA);
    descriptor.compute(resized,ret);
    return ret;
}
/**
 * Creates a new SVM model and trains it on the provided dataset. Then copies the support vectors of the obtained SVM to the internal one available for the HOGDescriptor.
*/
void Detector::train()
{
    if(accumulator.size()>0)
    {
        std::vector<cv::Mat> trainingDescriptors;
        std::vector<int> trainingLabels;
        computeImagesFeatures(); //--> could be done before and not inside, to decide
        for(HogTemplate& entry: accumulator)
        {
            cv::Mat tmp = cv::Mat(entry.descriptors.size(),1,CV_32FC1,entry.descriptors.data());
            trainingDescriptors.push_back(tmp.clone());
            trainingLabels.push_back(entry.label);
        }
        cv::Mat trainData;
        convert_to_ml(trainingDescriptors,trainData);
        svm->train(trainData,cv::ml::ROW_SAMPLE,cv::Mat(trainingLabels.size(),1,CV_32SC1,trainingLabels.data()));
    }
    cv::Mat suppVecs = svm->getSupportVectors();
    const int sv_total = suppVecs.rows;
    cv::Mat alpha, suppVecIdx;
    double rho = svm->getDecisionFunction(0, alpha, suppVecIdx);
    CV_Assert(alpha.total() == 1 && suppVecIdx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
              (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(suppVecs.type() == CV_32F);
    std::vector<float> hog_detector(suppVecs.cols + 1);
    memcpy(&hog_detector[0], suppVecs.ptr(), suppVecs.cols * sizeof(hog_detector[0]));
    hog_detector[suppVecs.cols] = (float)-rho;
    descriptor.setSVMDetector(hog_detector);
}
/**
 * Test the trained detector on a single provided image .
 * @param img The image to be used for the test.
 * @return Calculated bounding boxes (rectangles) and corresponding scores
*/
std::vector<Box> Detector::testImage(cv::Mat img)
{
    if(preprocess) preprocess(img);
    std::vector<cv::Rect> foundRects;
    std::vector<double> foundWeights;
    //Use of the provided method for detection of bounding rectangles
    descriptor.detectMultiScale(img,foundRects,foundWeights,0.2,cv::Size(4,4));
    std::vector<Box> ret;
    //Creation of the needed output by using the the Box struct.
    for (int i=0;i<foundRects.size();i++)
    {
        ret.push_back(Box(foundRects[i],foundWeights[i]));
    }
    return ret;
}

/**
 * Saves the model on a file.
 * @param filename name of the file to be created
*/
void Detector::save(std::string filename)
{
    descriptor.save(filename);
}
/**
 * Loads the model from a file.
 * @param filename name of the file to be used for loading
 * @return result of the loading procedure
*/
bool Detector::load(std::string filename)
{
    if(std::filesystem::exists(filename))
    {
        descriptor.load(filename);
        return true;
    }
    return false;
}
/**
 * Test the trained detector on a set of images called test set.
 * @param testImgs The images to be used as the test set.
*/
void Detector::testImages(std::vector<cv::Mat> testImgs)
{
    //clear total boxes found
    totalFoundBoxes.clear();
    std::vector<Box> temp;
    for (int i = 0; i < testImgs.size(); i++)
    {
        cv::Mat img;
        if (i < testImgs.size())
        {
            img = testImgs[i];
        }
        if (img.empty())
        {
            return;
        }
        temp.clear();
        temp = this->testImage(img);
        totalFoundBoxes.push_back((temp));
    }
}

/**
 * Function to convert the extracted features in order to be used as training data for the models
 * @param train_descs
 * @param trainData
*/
void Detector::convert_to_ml(const std::vector<cv::Mat>& train_descs, cv::Mat& trainData)
{
    const int rows = (int)train_descs.size();
    const int cols = (int)std::max(train_descs[0].cols, train_descs[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    for (size_t i = 0; i < train_descs.size(); ++i)
    {
        CV_Assert(train_descs[i].cols == 1 || train_descs[i].rows == 1);
        if (train_descs[i].cols == 1)
        {
            transpose(train_descs[i], tmp);
            tmp.copyTo(trainData.row((int)i));
        }
        else if (train_descs[i].rows == 1)
        {
            train_descs[i].copyTo(trainData.row((int)i));
        }
    }
}

/**
 * Retrieves the predicted bounding rectangles
 * @returns The predicted bounding rectangles for the images of the test set
 */
std::vector<std::vector<cv::Rect>> Detector::getRects()
{
    std::vector<std::vector<cv::Rect>> ret;
    std::vector<cv::Rect> temp;
    for(const std::vector<Box>& box_vect : totalFoundBoxes)
    {
        temp.clear();
        for(const Box& box : box_vect)
        {
             temp.push_back(box.bounding_box);
        }
        ret.push_back((temp));
    }
    return ret;
}

/**
 * Retrieves the confidence scores of the predicted rectangles
 * @returns The confidence scores of the predicted rectangles for the images of the test set
 */
std::vector<std::vector<double>> Detector::getConfidenceScores()
{
    std::vector<std::vector<double>> ret;
    std::vector<double> temp;
    for(const std::vector<Box>& box_vect : totalFoundBoxes)
    {
        temp.clear();
        for(const Box& box : box_vect)
        {
             temp.push_back(box.weight);
        }
        ret.push_back((temp));
    }
    return ret;
}