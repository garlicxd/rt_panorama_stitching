#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);

    cv::putText(img, "Hello OpenCV", {50,200},
    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
    cv::imshow("Demo", img);
    cv::waitKey(0);
    return 0;
}