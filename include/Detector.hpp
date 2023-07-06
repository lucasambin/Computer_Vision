/**
    Declaration of a class that performs the detection of objects based on the previously extracted features
    @file Detection.hpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#ifndef DETECTION_H
#define DETECTION_H
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

struct Template{
    std::string filename;
    int label;
    Template();
    Template(std::string filename,int label)
    {
        this->filename=filename;
        this->label=label;
    }
};
struct HogTemplate{
    std::string filename;
    int label;
    std::vector<float> descriptors;
    HogTemplate();
    HogTemplate(std::string filename,int label)
    {
        this->filename=filename;
        this->label=label;
    }
};
struct Box{
    cv::Rect bounding_box;
    double weight;
    Box(cv::Rect bounding_box,double weight)
    {
        this->bounding_box = bounding_box;
        this->weight = weight;
    }
};
class Detector
{
    private:
        std::vector<std::vector<cv::Rect>> totFoundRects;
        std::vector<std::vector<double>> totConfScores;
        cv::HOGDescriptor descriptor;
        cv::Size res_img_size;
        cv::Size winSize;
        cv::Size blockSize;
        cv::Size blockStride;
        cv::Size cellSize;
        int nbins;
        int derivAperture;
        int sampleSize;
        //
        std::vector<Template> templates;
        std::vector<HogTemplate> accumulator;
        std::vector<std::vector<Box>> totalFoundBoxes;
        //
        cv::Ptr<cv::ml::SVM> svm;
        cv::Mat (*preprocess)(cv::Mat&);
        //
        //
        void convert_to_ml(const std::vector<cv::Mat>& train_descs, cv::Mat& trainData);
        std::vector<std::string> readTrainingFiles(std::string dirname);
        std::vector<float> getHogForImage(cv::Mat img);
        void computeImagesFeatures();

    // Methods
    public:
        Detector(cv::Size box_size, cv::Size resize_image_size, cv::Mat (*preprocessing)(cv::Mat&) = nullptr);
        std::vector<HogTemplate> getAccumulator() const {return accumulator;}
        void addTrainingSet(std::string dirname, std::vector<int> labels);
        void addTrainingSet(std::string dirname, int label);
        void addTrainingSet(std::string dirname);
        void train();
        void save(std::string filename);
        bool load(std::string filename);
        std::vector<Box> testImage(cv::Mat img);
        void testImages(std::vector<cv::Mat> testImgs);
        std::vector<std::vector<cv::Rect>> getRects();
        std::vector<std::vector<double>> getConfidenceScores();
};

#endif