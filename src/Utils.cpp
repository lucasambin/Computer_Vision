/**
    Definition of a class that performs all utilities operations
    @file Utils.cpp
    @author Luca Sambin, Giacomo Seno, Davide Roana
*/
#include "Utils.hpp"
#include <opencv2/opencv.hpp>
/**
 * Applies the first check for the RGB image provided in one of the paper used
 * @param r red intensity
 * @param g green intensity
 * @param b blue intensity
 * @returns Result of the first check on the RGB image's pixel
 */
bool RGB_CHECK1(uchar r, uchar g, uchar b)
{
    return ((r > 95) && (g > 40) && (b > 20) && ((std::max({ r, g, b }) - std::min({ r, g, b })) > 15) && (std::abs(r - g) > 15) && (r > g) && (r > b));
}
/**
 * Applies the second check for the RGB image provided in one of the paper used
 * @param r red intensity
 * @param g green intensity
 * @param b blue intensity
 * @returns Result of the second check on the RGB image's pixel
 */
bool RGB_CHECK2(uchar r, uchar g, uchar b)
{
    return ((r > 220) && (g > 210) && (b > 170) && (std::abs(r - g) <= 15) && (r > b) && (g > b));
}
/**
 * Applies the overall check for the RGB image provided in one of the paper used
 * @param r red intensity
 * @param g green intensity
 * @param b blue intensity
 * @returns Result of the overall check on the RGB image's pixel
 */
bool RULEA(uchar r, uchar g, uchar b)
{
    return RGB_CHECK1(r, g, b) || RGB_CHECK2(r, g, b);
}
/**
 * Applies the check for the yCrCb image provided in one of the paper used
 * @param cr cr value of the yCrCb image's pixel
 * @param cb cb value of the yCrCb image's pixel
 * @returns Result of the overall check on the yCrCb image's pixel
 */
bool CRCB_CHECK(uchar cr, uchar cb)
{
    return ((cr <= (1.5862 * cb + 20)) && (cr >= (0.3448 * cb + 76.2069)) && (cr >= (-4.5652 * cb + 234.5652)) && (cr <= (-1.15 * cb + 301.75)) && (cr <= (-2.2857 * cb + 432.85)));
}
/**
 * Applies the check for the hsv image provided in one of the paper used
 * @param h h value of the hsv image's pixel
 * @returns Result of the overall check on the hsv image's pixel
 */
bool HSV_CHECK(uchar h)
{
    return ((h < 18) || (h > 161));
}
/**
 * Applies the overall check of the color provided in one of the paper used
 * @param r red intensity
 * @param g green intensity
 * @param b blue intensity
 * @param cr cr value of the yCrCb image's pixel
 * @param cb cb value of the yCrCb image's pixel
 * @param h h value of the hsv image's pixel
 * @returns Result of the overall check on of the color on the image's pixel
 */
bool pixel_color_check(uchar r, uchar g, uchar b, uchar cr, uchar cb, uchar h)
{
    return RULEA(r, g, b) && CRCB_CHECK(cr, cb) && HSV_CHECK(h);
}
/**
 * Applies the overall check of the color and provides a mask as the result of the check
 * @param src image to apply the color threshold to to obtain the mask
 * @param boxes bounding boxes to apply the check inside of (Regions of Interest)
 * @param type type of the check, either based on the first or second paper cited
 * @returns mask obtained
 */
cv::Mat mask_color_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes, int type)
{
    if (type != SKIN_TYPE_1 && type != SKIN_TYPE_2)
    {
        std::cout << "Invalid argument: " << type << "!";
        exit(-1);
    }
    cv::Mat image = src.clone();
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    for (int i = 0; i < boxes.size(); i++)
    {
        // ROI of original image
        cv::Mat img = image(boxes[i]);
        int xoff = boxes[i].x;
        int yoff = boxes[i].y;

        cv::Mat filtered_rgb = img.clone();
        cv::Mat filtered_ycrcb;
        cv::cvtColor(img, filtered_ycrcb, cv::COLOR_BGR2YCrCb);
        cv::Mat filtered_hsv;
        cv::cvtColor(img, filtered_hsv, cv::COLOR_BGR2HSV);

        //filtering in rgb
        cv::Vec3b rgb_i;
        cv::Vec3b ycrcb_i;
        cv::Vec3b hsv_i;
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                rgb_i = filtered_rgb.at<cv::Vec3b>(i, j);
                ycrcb_i = filtered_ycrcb.at<cv::Vec3b>(i, j);
                hsv_i = filtered_hsv.at<cv::Vec3b>(i, j);
                if (type == SKIN_TYPE_2)
                {
                    float cr = ycrcb_i[1];
                    float cb = ycrcb_i[2];
                    uchar r = rgb_i.val[2];
                    uchar g = rgb_i.val[1];
                    uchar b = rgb_i.val[0];
                    uchar h = hsv_i.val[0] * 2;
                    float s = hsv_i.val[1] / 255.;
                    if (((r > 95) && (g > 40) && (b > 20) && (std::abs(r - g) > 15) && (r > g) && (r > b) && (h >= 0) && (h <= 50) && (s >= 0.23) && (s <= 0.68)) || ((r > 95) && (g > 40) && (b > 20) && (std::abs(r - g) > 15) && (r > g) && (r > b) && (cr > 135) && (cb > 85) && (cr >= (0.3448 * cb) + 76.2069) && (cr >= (-4.5652 * cb) + 234.5652) && (cr <= (-1.15 * cb) + 301.75) && (cr <= (-2.2857 * cb) + 432.85)))
                    {
                        mask.at<uchar>(yoff + i, xoff + j) = 255;
                    }
                }
                else if (type == SKIN_TYPE_1)
                {
                    if (pixel_color_check(rgb_i.val[2], rgb_i.val[1], rgb_i.val[0], ycrcb_i.val[1], ycrcb_i.val[2], hsv_i.val[0]))
                    {
                        mask.at<uchar>(yoff + i, xoff + j) = 255;
                    }
                }
            }
        }
    }
    cv::bitwise_not(mask, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 5);
    cv::bitwise_not(mask, mask);
    return mask;
}

/**
 * Applies the mask on the provided image
 * @param src image to apply the mask to
 * @param mask mask to be applied to the image
 */
void mask_image(cv::Mat& src, cv::Mat mask)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (mask.at<uchar>(i, j) == 0)
            {
                cv::Vec3b& intensity = src.at<cv::Vec3b>(i, j);
                intensity[0] = 0;
                intensity[1] = 0;
                intensity[2] = 0;
            }
        }
    }
}
/**
 * Applies the color threshold on the provided image, setting to black all pixels not deemed to be similar to skin color
 * @param src image to apply gthe threshold to
 * @return mask applied on the image
 */
cv::Mat apply_skin_threshold(cv::Mat& src)
{
    std::vector<cv::Rect> boxes;
    boxes.push_back(cv::Rect(0, 0, src.cols, src.rows));
    cv::Mat mask = mask_skin_threshold(src, boxes);
    mask_image(src, mask);
    return mask;
}
/**
 * Applies the color threshold on the provided image and boxes provided with default type, so based on the first paper
 * @param src image to apply the threshold to to obtain the mask
 * @param boxes bounding boxes to apply the check inside of (Regions of Interest)
 * @return mask obtained on the image
 */
cv::Mat mask_skin_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes)
{
    return mask_color_threshold(src, boxes, SKIN_TYPE_1);
}
/**
 * Applies the color threshold on the provided image and boxes provided
 * @param src image to apply the threshold to to obtain the mask
 * @param boxes bounding boxes to apply the check inside of (Regions of Interest)
 * @param type type of the check, either based on the first or second paper cited
 * @return mask obtained on the image
 */
cv::Mat mask_skin_threshold(const cv::Mat& src, const std::vector<cv::Rect>& boxes, int type)
{
    return mask_color_threshold(src, boxes, type);
}
