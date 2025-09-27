#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <string>
#include <map>
#include <numeric>

// config
const bool USE_SIFT = true;
const bool USE_FEATHERING = true;
const double RATIO_THRESHOLD = 0.8;
const double RANSAC_THRESHOLD = 5.0;

const int MIN_GOOD_MATCHES = 20;
const double MIN_INLIER_RATIO = 0.20;
const int MIN_ABSOLUTE_INLIERS = 8;

const bool DISABLE_QUALITY_CONTROL = false;

const bool RUN_PAIR_EXPERIMENTS = false;
const bool RUN_FULL_PANORAMA_EXPERIMENTS = false;

const bool RUN_EXPERIMENTS = false;
const std::vector<double> RANSAC_THRESHOLDS = {1.0, 3.0, 5.0, 7.0, 10.0};

const int MAX_SIFT_FEATURES = 800;
const int MAX_ORB_FEATURES = 500;
const int MAX_PANORAMA_WIDTH = 20000;
const int MAX_PANORAMA_HEIGHT = 20000;
const int MAX_INPUT_SIZE = 1500;
const double COMPRESSION_THRESHOLD = 0.6;

// a struct
struct FeatureMatchingResults {
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> matched_points1, matched_points2;
    std::vector<cv::DMatch> all_matches;
    std::vector<float> match_distances;
    double matching_time_ms;
    double detection_time_ms;
    int initial_keypoints1, initial_keypoints2;
    std::string detector_name;
};

struct HomographyResults {
    cv::Mat homography;
    std::vector<uchar> inliers_mask;
    int num_inliers;
    double ransac_time_ms;
};

struct ExperimentalResults {
    std::string detector_name;
    std::string blend_method;
    double ransac_threshold;
    int keypoints1, keypoints2;
    int good_matches_count;
    int inliers_count;
    double inlier_ratio;
    double detection_time_ms;
    double matching_time_ms;
    double ransac_time_ms;
    double total_time_ms;
    std::vector<float> match_distances;
    cv::Size panorama_size;
};

cv::Mat resizeImageIfNeeded(const cv::Mat& image, double& scaleFactor) {
    scaleFactor = 1.0;
    if (image.empty()) return image;
    
    int maxDim = std::max(image.cols, image.rows);
    if (maxDim > MAX_INPUT_SIZE) {
        scaleFactor = static_cast<double>(MAX_INPUT_SIZE) / maxDim;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR); // scale it down
        std::cout << "  Resized from " << image.cols << "x" << image.rows 
                  << " to " << resized.cols << "x" << resized.rows 
                  << " (scale: " << std::fixed << std::setprecision(2) << scaleFactor << ")" << std::endl;
        return resized;
    }
    return image;
}

bool isPanoramaTooLarge(const cv::Mat& panorama) {
    if (panorama.empty()) return false;
    return (panorama.cols > MAX_PANORAMA_WIDTH || panorama.rows > MAX_PANORAMA_HEIGHT);
}

cv::Mat compressPanoramaIfNeeded(const cv::Mat& panorama, double& compressionFactor, int currentImage, int totalImages) {
    compressionFactor = 1.0;
    
    if (panorama.empty()) return panorama;
    
    int remainingImages = totalImages - currentImage;
    double aggressiveness = 1.0 + (remainingImages * 0.1); // this gets bigger the more images are left
    
    double threshold = COMPRESSION_THRESHOLD / aggressiveness;
    double widthRatio = static_cast<double>(panorama.cols) / (MAX_PANORAMA_WIDTH * threshold);
    double heightRatio = static_cast<double>(panorama.rows) / (MAX_PANORAMA_HEIGHT * threshold);
    double maxRatio = std::max(widthRatio, heightRatio);
    
    if (maxRatio > 1.0) {
        compressionFactor = 0.85 / maxRatio;
        
        cv::Mat compressed;
        cv::resize(panorama, compressed, cv::Size(0, 0), compressionFactor, compressionFactor, cv::INTER_LINEAR);
        
        std::cout << "  Compressed panorama from " << panorama.cols << "x" << panorama.rows 
                  << " to " << compressed.cols << "x" << compressed.rows 
                  << " (scale: " << std::fixed << std::setprecision(3) << compressionFactor 
                  << ", remaining: " << remainingImages << " images)" << std::endl;
        
        return compressed;
    }
    
    return panorama;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> detectAndDescribe(const cv::Mat& image, bool use_sift = USE_SIFT) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    if (use_sift) {
        try {
            auto detector = cv::SIFT::create(MAX_SIFT_FEATURES);
            detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        } catch (const cv::Exception& e) {
            std::cout << "SIFT not available, falling back to ORB: " << e.what() << std::endl;
            auto detector = cv::ORB::create(MAX_ORB_FEATURES);
            detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        }
    } else {
        auto detector = cv::ORB::create(MAX_ORB_FEATURES);
        detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }
    
    return std::make_pair(keypoints, descriptors);
}

FeatureMatchingResults matchFeatures(const cv::Mat& img1, const cv::Mat& img2, bool use_sift = USE_SIFT) {
    FeatureMatchingResults results;
    results.detector_name = use_sift ? "SIFT" : "ORB";
    
    auto start_detection = std::chrono::high_resolution_clock::now();
    
    double scale1, scale2;
    cv::Mat img1_resized = resizeImageIfNeeded(img1, scale1);
    cv::Mat img2_resized = resizeImageIfNeeded(img2, scale2);
    
    auto [keypoints1, descriptors1] = detectAndDescribe(img1_resized, use_sift);
    auto [keypoints2, descriptors2] = detectAndDescribe(img2_resized, use_sift);
    
    if (scale1 != 1.0) {
        for (auto& kp : keypoints1) {
            kp.pt.x /= scale1;
            kp.pt.y /= scale1;
        }
    }
    if (scale2 != 1.0) {
        for (auto& kp : keypoints2) {
            kp.pt.x /= scale2;
            kp.pt.y /= scale2;
        }
    }
    
    auto end_detection = std::chrono::high_resolution_clock::now();
    results.detection_time_ms = std::chrono::duration<double, std::milli>(end_detection - start_detection).count();
    
    results.keypoints1 = keypoints1;
    results.keypoints2 = keypoints2;
    results.initial_keypoints1 = static_cast<int>(keypoints1.size());
    results.initial_keypoints2 = static_cast<int>(keypoints2.size());
    
    if (descriptors1.empty() || descriptors2.empty()) {
        std::cout << "Warning: No descriptors found in one or both images!" << std::endl;
        return results;
    }
    
    auto start_matching = std::chrono::high_resolution_clock::now();
    
    cv::BFMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
    if (use_sift) {
        matcher = cv::BFMatcher(cv::NORM_L2);
    } else {
        matcher = cv::BFMatcher(cv::NORM_HAMMING);
    }
    
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2); // find neighbors
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2) {
            if (match_pair[0].distance < RATIO_THRESHOLD * match_pair[1].distance) { // the ratio test
                results.good_matches.push_back(match_pair[0]);
                results.matched_points1.push_back(keypoints1[match_pair[0].queryIdx].pt);
                results.matched_points2.push_back(keypoints2[match_pair[0].trainIdx].pt);
                results.match_distances.push_back(match_pair[0].distance);
            }
            results.all_matches.push_back(match_pair[0]);
        }
    }
    
    auto end_matching = std::chrono::high_resolution_clock::now();
    results.matching_time_ms = std::chrono::duration<double, std::milli>(end_matching - start_matching).count();
    
    return results;
}

HomographyResults estimateHomography(const std::vector<cv::Point2f>& points1, 
                                    const std::vector<cv::Point2f>& points2,
                                    double threshold = RANSAC_THRESHOLD) {
    HomographyResults results;
    
    if (points1.size() < 4 || points2.size() < 4) {
        std::cout << "Error: Need at least 4 point correspondences for homography estimation!" << std::endl;
        return results;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // note the order: this finds the transform from image 2's points to image 1's points
    results.homography = cv::findHomography(points2, points1, cv::RANSAC,
                                          threshold, results.inliers_mask);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.ransac_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    results.num_inliers = cv::sum(results.inliers_mask)[0];
    
    return results;
}

cv::Mat stitchTwoImagesOptimized(const cv::Mat& img1, const cv::Mat& img2) {
    if (isPanoramaTooLarge(img1)) {
        std::cout << "Warning: First image already too large, skipping..." << std::endl;
        return cv::Mat();
    }
    
    FeatureMatchingResults match_results = matchFeatures(img1, img2);
    
    if (match_results.good_matches.size() < MIN_GOOD_MATCHES) {
        std::cout << "Error: Not enough good matches (" << match_results.good_matches.size() 
                  << ") - need at least " << MIN_GOOD_MATCHES << " for reliable stitching!" << std::endl;
        return cv::Mat();
    }
    
    std::cout << "  Keypoints: " << match_results.initial_keypoints1 << " + " << match_results.initial_keypoints2 
              << ", Good matches: " << match_results.good_matches.size() << std::endl;
    
    HomographyResults homo_results = estimateHomography(match_results.matched_points1, 
                                                       match_results.matched_points2);
    
    if (homo_results.homography.empty()) {
        std::cout << "Error: Failed to estimate homography!" << std::endl;
        return cv::Mat();
    }
    
    double inlier_ratio = static_cast<double>(homo_results.num_inliers) / match_results.good_matches.size();
    
    std::cout << "  RANSAC inliers: " << homo_results.num_inliers << " (" 
              << std::fixed << std::setprecision(1) << (100.0 * inlier_ratio) << "%)" << std::endl;
    
    if (!DISABLE_QUALITY_CONTROL && (inlier_ratio < MIN_INLIER_RATIO || homo_results.num_inliers < MIN_ABSOLUTE_INLIERS)) {
        std::cout << "Warning: Poor homography quality - inlier ratio: " << std::fixed << std::setprecision(1) 
                  << (100.0 * inlier_ratio) << "% (need >" << (MIN_INLIER_RATIO * 100) 
                  << "%) or inliers: " << homo_results.num_inliers << " (need >" << MIN_ABSOLUTE_INLIERS << ")" << std::endl;
        std::cout << "Skipping this image pair due to poor match quality." << std::endl;
        return cv::Mat();
    }
    
    if (DISABLE_QUALITY_CONTROL && (inlier_ratio < MIN_INLIER_RATIO || homo_results.num_inliers < MIN_ABSOLUTE_INLIERS)) {
        std::cout << "WARNING: Quality control BYPASSED - proceeding with poor quality homography!" << std::endl;
        std::cout << "         Inlier ratio: " << std::fixed << std::setprecision(1) << (100.0 * inlier_ratio) 
                  << "%, Inliers: " << homo_results.num_inliers << std::endl;
    }
    
    std::vector<cv::Point2f> corners = {
        {0, 0}, 
        {static_cast<float>(img2.cols), 0}, 
        {static_cast<float>(img2.cols), static_cast<float>(img2.rows)}, 
        {0, static_cast<float>(img2.rows)}
    };
    
    std::vector<cv::Point2f> warped_corners;
    cv::perspectiveTransform(corners, warped_corners, homo_results.homography);
    
    float min_x = 0, min_y = 0, max_x = static_cast<float>(img1.cols), max_y = static_cast<float>(img1.rows);
    for (const auto& corner : warped_corners) {
        min_x = std::min(min_x, corner.x);
        min_y = std::min(min_y, corner.y);
        max_x = std::max(max_x, corner.x);
        max_y = std::max(max_y, corner.y);
    }
    
    int output_width = static_cast<int>(std::ceil(max_x - min_x));
    int output_height = static_cast<int>(std::ceil(max_y - min_y));
    
    if (output_width > MAX_PANORAMA_WIDTH || output_height > MAX_PANORAMA_HEIGHT) {
        std::cout << "Warning: Resulting panorama would be too large (" << output_width << "x" << output_height 
                  << "), skipping..." << std::endl;
        return cv::Mat();
    }
    
    // this moves the canvas so the stitched image doesn't have negative coordinates
    cv::Mat translation = (cv::Mat_<double>(3, 3) << 
        1, 0, -min_x,
        0, 1, -min_y,
        0, 0, 1);
    
    cv::Mat adjusted_homography = translation * homo_results.homography;
    
    cv::Mat img2_warped, mask;
    cv::warpPerspective(img2, img2_warped, adjusted_homography, 
                       cv::Size(output_width, output_height));
    cv::warpPerspective(cv::Mat::ones(img2.rows, img2.cols, CV_8UC1), mask, 
                       adjusted_homography, cv::Size(output_width, output_height));
    
    cv::Mat img1_on_canvas = cv::Mat::zeros(output_height, output_width, img1.type());
    int offset_x = static_cast<int>(-min_x);
    int offset_y = static_cast<int>(-min_y);
    
    if (offset_x >= 0 && offset_y >= 0 && 
        offset_x + img1.cols <= output_width && 
        offset_y + img1.rows <= output_height) {
        img1.copyTo(img1_on_canvas(cv::Rect(offset_x, offset_y, img1.cols, img1.rows)));
    }
    
    cv::Mat result = img1_on_canvas.clone();
    img2_warped.copyTo(result, mask);
    
    return result;
}

std::vector<std::string> getImageFiles(const std::string& directory) {
    std::vector<std::string> imageFiles;
    
    if (!std::filesystem::exists(directory)) {
        std::cout << "Directory does not exist: " << directory << std::endl;
        return imageFiles;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();
            
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                if (filename.find("-min.jpg") != std::string::npos) {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }
    }
    
    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

std::string getDatasetName(const std::string& path) {
    std::filesystem::path p(path);
    return p.filename().string();
}

// the main stitcher
cv::Mat stitchMultipleImagesOptimized(const std::vector<std::string>& imagePaths, 
                                     const std::string& datasetName) {
    if (imagePaths.size() < 2) {
        std::cout << "Need at least 2 images for stitching!" << std::endl;
        return cv::Mat();
    }
    
    std::cout << "=== PERFORMANCE-OPTIMIZED SEQUENTIAL STITCHING ===" << std::endl;
    std::cout << "Dataset: " << datasetName << std::endl;
    std::cout << "Images to stitch: " << imagePaths.size() << std::endl;
    
    cv::Mat panorama = cv::imread(imagePaths[0]);
    if (panorama.empty()) {
        std::cout << "Error loading first image: " << imagePaths[0] << std::endl;
        return cv::Mat();
    }
    
    std::cout << "Starting with: " << std::filesystem::path(imagePaths[0]).filename().string() << std::endl;
    
    // main loop
    for (size_t i = 1; i < imagePaths.size(); ++i) {
        cv::Mat nextImage = cv::imread(imagePaths[i]);
        if (nextImage.empty()) {
            std::cout << "Warning: Could not load image " << imagePaths[i] << ", skipping..." << std::endl;
            continue;
        }
        
        std::cout << "\nStitching with: " << std::filesystem::path(imagePaths[i]).filename().string() << std::endl;
        
        cv::Mat newPanorama = stitchTwoImagesOptimized(panorama, nextImage);
        
        if (newPanorama.empty()) {
            std::cout << "Warning: Failed to stitch image " << i << ", trying to continue with next image..." << std::endl;
            nextImage.release();
            continue;
        }
        
        panorama = newPanorama;
        std::cout << "Current panorama size: " << panorama.cols << "x" << panorama.rows << std::endl;
        
        if (imagePaths.size() > 2) {
            double compressionFactor = 1.0;
            cv::Mat compressedPanorama = compressPanoramaIfNeeded(panorama, compressionFactor, i, imagePaths.size());
            
            if (compressionFactor < 1.0) {
                panorama = compressedPanorama;
                std::cout << "  Compressed panorama size: " << panorama.cols << "x" << panorama.rows << std::endl;
            }
        }
        
        newPanorama.release();
        nextImage.release();
    }
    
    return panorama;
}

// save results
void saveExperimentalDataCSV(const std::vector<ExperimentalResults>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    file << "detector_name,blend_method,ransac_threshold,keypoints1,keypoints2,good_matches,inliers,inlier_ratio,"
         << "detection_time_ms,matching_time_ms,ransac_time_ms,total_time_ms,panorama_width,panorama_height,avg_match_distance\n";
    
    for (const auto& result : results) {
        double avgDist = 0.0;
        if (!result.match_distances.empty()) {
            avgDist = std::accumulate(result.match_distances.begin(), result.match_distances.end(), 0.0) / result.match_distances.size();
        }
        
        file << result.detector_name << ","
             << result.blend_method << ","
             << result.ransac_threshold << ","
             << result.keypoints1 << ","
             << result.keypoints2 << ","
             << result.good_matches_count << ","
             << result.inliers_count << ","
             << std::fixed << std::setprecision(4) << result.inlier_ratio << ","
             << result.detection_time_ms << ","
             << result.matching_time_ms << ","
             << result.ransac_time_ms << ","
             << result.total_time_ms << ","
             << result.panorama_size.width << ","
             << result.panorama_size.height << ","
             << avgDist << "\n";
    }
    file.close();
    std::cout << "Experimental data saved to: " << filename << std::endl;
}

void saveExperimentalDataJSON(const std::vector<ExperimentalResults>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "{\n  \"experiments\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        double avgDist = 0.0;
        if (!result.match_distances.empty()) {
            avgDist = std::accumulate(result.match_distances.begin(), result.match_distances.end(), 0.0) / result.match_distances.size();
        }
        
        file << "    {\n"
             << "      \"detector_name\": \"" << result.detector_name << "\",\n"
             << "      \"blend_method\": \"" << result.blend_method << "\",\n"
             << "      \"ransac_threshold\": " << result.ransac_threshold << ",\n"
             << "      \"keypoints1\": " << result.keypoints1 << ",\n"
             << "      \"keypoints2\": " << result.keypoints2 << ",\n"
             << "      \"good_matches\": " << result.good_matches_count << ",\n"
             << "      \"inliers\": " << result.inliers_count << ",\n"
             << "      \"inlier_ratio\": " << std::fixed << std::setprecision(4) << result.inlier_ratio << ",\n"
             << "      \"detection_time_ms\": " << result.detection_time_ms << ",\n"
             << "      \"matching_time_ms\": " << result.matching_time_ms << ",\n"
             << "      \"ransac_time_ms\": " << result.ransac_time_ms << ",\n"
             << "      \"total_time_ms\": " << result.total_time_ms << ",\n"
             << "      \"panorama_size\": {\"width\": " << result.panorama_size.width << ", \"height\": " << result.panorama_size.height << "},\n"
             << "      \"avg_match_distance\": " << avgDist << ",\n"
             << "      \"match_distances\": [";
        
        for (size_t j = 0; j < result.match_distances.size() && j < 100; ++j) {
            file << result.match_distances[j];
            if (j < result.match_distances.size() - 1 && j < 99) file << ", ";
        }
        
        file << "]\n    }";
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n}\n";
    file.close();
    std::cout << "Experimental data saved to: " << filename << std::endl;
}

cv::Mat visualizeKeypoints(const cv::Mat& img1, const cv::Mat& img2, 
                          const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                          const std::vector<cv::DMatch>& matches) {
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches, // pretty lines
                   cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    std::string stats = "Keypoints: " + std::to_string(kp1.size()) + " + " + std::to_string(kp2.size()) + 
                       " | Matches: " + std::to_string(matches.size());
    cv::putText(img_matches, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    return img_matches;
}

void saveVisualization(const cv::Mat& img1, const cv::Mat& img2,
                      const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                      const std::vector<cv::DMatch>& matches, const std::string& filename) {
    cv::Mat visualization = visualizeKeypoints(img1, img2, kp1, kp2, matches);
    cv::imwrite(filename, visualization);
    std::cout << "Visualization saved to: " << filename << std::endl;
}

// make a graph
cv::Mat plotHistogram(const std::vector<float>& data, const std::string& title, int bins = 50) {
    if (data.empty()) {
        cv::Mat empty(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::putText(empty, "No data to plot", cv::Point(200, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        return empty;
    }
    
    auto minmax = std::minmax_element(data.begin(), data.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;
    
    std::vector<int> histogram(bins, 0);
    float range = maxVal - minVal;
    
    for (float value : data) {
        int bin = static_cast<int>(((value - minVal) / range) * (bins - 1));
        bin = std::max(0, std::min(bins - 1, bin));
        histogram[bin]++;
    }
    
    int maxCount = *std::max_element(histogram.begin(), histogram.end());
    
    int width = 800;
    int height = 600;
    int margin = 80;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    int barWidth = (width - 2 * margin) / bins;
    for (int i = 0; i < bins; ++i) {
        int barHeight = static_cast<int>(((double)histogram[i] / maxCount) * (height - 2 * margin));
        cv::Point p1(margin + i * barWidth, height - margin);
        cv::Point p2(margin + (i + 1) * barWidth, height - margin - barHeight);
        cv::rectangle(plot, p1, p2, cv::Scalar(100, 150, 255), -1);
        cv::rectangle(plot, p1, p2, cv::Scalar(0, 0, 0), 1);
    }
    
    cv::putText(plot, title, cv::Point(width/2 - title.length() * 8, 40), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(plot, "Match Distance", cv::Point(width/2 - 60, height - 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(plot, "Frequency", cv::Point(10, height/2), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(plot, std::to_string(minVal).substr(0, 5), cv::Point(margin - 20, height - margin + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(plot, std::to_string(maxVal).substr(0, 5), cv::Point(width - margin - 30, height - margin + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    return plot;
}

void saveHistogram(const std::vector<float>& data, const std::string& title, const std::string& filename) {
    cv::Mat histogram = plotHistogram(data, title);
    cv::imwrite(filename, histogram);
    std::cout << "Histogram saved to: " << filename << std::endl;
}

void printMetricsTable(const std::vector<ExperimentalResults>& results) {
    std::cout << "\n=== QUANTITATIVE METRICS TABLE ===" << std::endl;
    std::cout << std::left << std::setw(12) << "Detector"
              << std::setw(12) << "Blend"
              << std::setw(8) << "RANSAC"
              << std::setw(8) << "KP1"
              << std::setw(8) << "KP2"
              << std::setw(8) << "Matches"
              << std::setw(8) << "Inliers"
              << std::setw(8) << "In.Ratio"
              << std::setw(12) << "Detect(ms)"
              << std::setw(12) << "Match(ms)"
              << std::setw(12) << "RANSAC(ms)"
              << std::setw(12) << "Total(ms)" << std::endl;
    
    std::cout << std::string(140, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(12) << result.detector_name
                  << std::setw(12) << result.blend_method
                  << std::setw(8) << std::fixed << std::setprecision(1) << result.ransac_threshold
                  << std::setw(8) << result.keypoints1
                  << std::setw(8) << result.keypoints2
                  << std::setw(8) << result.good_matches_count
                  << std::setw(8) << result.inliers_count
                  << std::setw(8) << std::fixed << std::setprecision(3) << result.inlier_ratio
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.detection_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.matching_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.ransac_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time_ms << std::endl;
    }
    std::cout << std::string(140, '-') << std::endl;
}

std::vector<ExperimentalResults> runComprehensiveExperiments(const cv::Mat& img1, const cv::Mat& img2, 
                                                            const std::string& outputDir, 
                                                            const std::string& imagePairName) {
    std::vector<ExperimentalResults> allResults;
    std::filesystem::create_directories(outputDir);
    
    std::cout << "\n=== COMPREHENSIVE EXPERIMENTAL ANALYSIS ===" << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    
    std::vector<std::pair<bool, std::string>> detectorConfigs = {{true, "SIFT"}, {false, "ORB"}};
    std::vector<std::pair<bool, std::string>> blendConfigs = {{true, "Feathering"}, {false, "SimpleOverlay"}};
    
    for (const auto& detConfig : detectorConfigs) {
        for (const auto& blendConfig : blendConfigs) {
            for (double ransacThresh : RANSAC_THRESHOLDS) {
                std::cout << "\nTesting: " << detConfig.second << " + " << blendConfig.second 
                         << " (RANSAC: " << ransacThresh << ")" << std::endl;
                
                auto start = std::chrono::high_resolution_clock::now();
                
                // timing
                auto detectStart = std::chrono::high_resolution_clock::now();
                auto [kp1, desc1] = detectAndDescribe(img1, detConfig.first);
                auto [kp2, desc2] = detectAndDescribe(img2, detConfig.first);
                auto detectEnd = std::chrono::high_resolution_clock::now();
                
                if (desc1.empty() || desc2.empty()) {
                    std::cout << "Warning: No descriptors found, skipping..." << std::endl;
                    continue;
                }
                
                auto matchStart = std::chrono::high_resolution_clock::now();
                
                cv::BFMatcher matcher;
                std::vector<std::vector<cv::DMatch>> knn_matches;
                
                if (detConfig.first) {
                    matcher = cv::BFMatcher(cv::NORM_L2);
                } else {
                    matcher = cv::BFMatcher(cv::NORM_HAMMING);
                }
                
                matcher.knnMatch(desc1, desc2, knn_matches, 2);
                
                std::vector<cv::DMatch> good_matches;
                std::vector<cv::Point2f> matched_points1, matched_points2;
                std::vector<float> match_distances;
                
                for (const auto& match_pair : knn_matches) {
                    if (match_pair.size() >= 2) {
                        if (match_pair[0].distance < RATIO_THRESHOLD * match_pair[1].distance) {
                            good_matches.push_back(match_pair[0]);
                            matched_points1.push_back(kp1[match_pair[0].queryIdx].pt);
                            matched_points2.push_back(kp2[match_pair[0].trainIdx].pt);
                            match_distances.push_back(match_pair[0].distance);
                        }
                    }
                }
                
                auto matchEnd = std::chrono::high_resolution_clock::now();
                
                if (good_matches.empty()) {
                    std::cout << "Warning: No good matches found, skipping..." << std::endl;
                    continue;
                }
                
                auto ransacStart = std::chrono::high_resolution_clock::now();
                HomographyResults homResults = estimateHomography(matched_points1, matched_points2, ransacThresh);
                auto ransacEnd = std::chrono::high_resolution_clock::now();
                
                auto totalEnd = std::chrono::high_resolution_clock::now();
                
                ExperimentalResults result;
                result.detector_name = detConfig.second;
                result.blend_method = blendConfig.second;
                result.ransac_threshold = ransacThresh;
                result.keypoints1 = kp1.size();
                result.keypoints2 = kp2.size();
                result.good_matches_count = good_matches.size();
                result.inliers_count = homResults.num_inliers;
                result.inlier_ratio = static_cast<double>(homResults.num_inliers) / good_matches.size();
                result.detection_time_ms = std::chrono::duration<double, std::milli>(detectEnd - detectStart).count();
                result.matching_time_ms = std::chrono::duration<double, std::milli>(matchEnd - matchStart).count();
                result.ransac_time_ms = std::chrono::duration<double, std::milli>(ransacEnd - ransacStart).count();
                result.total_time_ms = std::chrono::duration<double, std::milli>(totalEnd - start).count();
                result.match_distances = match_distances;
                result.panorama_size = cv::Size(img1.cols + img2.cols, std::max(img1.rows, img2.rows));
                
                allResults.push_back(result);
                
                if (ransacThresh == 5.0) {
                    std::string configName = detConfig.second + "_" + blendConfig.second;
                    std::string visFile = outputDir + "/" + imagePairName + "_" + configName + "_matches.jpg";
                    saveVisualization(img1, img2, kp1, kp2, good_matches, visFile);
                    
                    std::string histFile = outputDir + "/" + imagePairName + "_" + configName + "_distances.jpg";
                    saveHistogram(match_distances, "Match Distances - " + configName, histFile);
                }
            }
        }
    }
    
    printMetricsTable(allResults);
    
    std::string csvFile = outputDir + "/" + imagePairName + "_experimental_data.csv";
    std::string jsonFile = outputDir + "/" + imagePairName + "_experimental_data.json";
    saveExperimentalDataCSV(allResults, csvFile);
    saveExperimentalDataJSON(allResults, jsonFile);
    
    return allResults;
}

void runPairStitchingExperiments(const std::vector<std::string>& imageFiles, const std::string& datasetName) {
    if (imageFiles.size() < 4) {
        std::cout << "Need at least 4 images for pair experiments (2nd + 4th image)" << std::endl;
        return;
    }
    
    std::cout << "\n=== PAIR STITCHING EXPERIMENTS (2nd + 4th images) ===" << std::endl;
    
    std::string pairOutputDir = "pair_stitching/" + datasetName;
    std::filesystem::create_directories(pairOutputDir);
    
    cv::Mat img2 = cv::imread(imageFiles[1]);
    cv::Mat img4 = cv::imread(imageFiles[3]);
    
    if (img2.empty() || img4.empty()) {
        std::cout << "Failed to load 2nd or 4th image" << std::endl;
        return;
    }
    
    std::cout << "Processing pair: " << std::filesystem::path(imageFiles[1]).filename().string() 
              << " + " << std::filesystem::path(imageFiles[3]).filename().string() << std::endl;
    
    std::string imagePairName = datasetName + "_pair_2_4";
    std::vector<ExperimentalResults> pairResults = runComprehensiveExperiments(img2, img4, pairOutputDir, imagePairName);
    
    std::cout << "Creating panorama from 2nd + 4th images..." << std::endl;
    cv::Mat pairPanorama = stitchTwoImagesOptimized(img2, img4);
    
    if (!pairPanorama.empty()) {
        std::string pairPanoramaPath = pairOutputDir + "/" + datasetName + "_pair_2_4_panorama.jpg";
        if (cv::imwrite(pairPanoramaPath, pairPanorama)) {
            std::cout << "Pair panorama saved: " << pairPanoramaPath << std::endl;
            std::cout << "Pair panorama size: " << pairPanorama.cols << "x" << pairPanorama.rows << std::endl;
        }
    } else {
        std::cout << "Failed to create panorama from 2nd + 4th images" << std::endl;
    }
    
    std::cout << "Pair stitching experiments completed for " << datasetName << std::endl;
}

void moveOutputFilesToFolder(const std::string& sourceDir, const std::string& destDir) {
    try {
        std::filesystem::create_directories(destDir);
        for (const auto& entry : std::filesystem::directory_iterator(sourceDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string destPath = destDir + "/" + filename;
                std::filesystem::copy_file(entry.path(), destPath, std::filesystem::copy_options::overwrite_existing);
                std::filesystem::remove(entry.path());
                std::cout << "  Moved " << filename << " to " << destDir << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Warning: Error moving files: " << e.what() << std::endl;
    }
}

// entry point
int main(int argc, char* argv[]) {
    std::cout << "=== PERFORMANCE-OPTIMIZED PANORAMA STITCHING ===" << std::endl;
    std::cout << "Performance limits:" << std::endl;
    std::cout << "  Max SIFT features: " << MAX_SIFT_FEATURES << std::endl;
    std::cout << "  Max ORB features: " << MAX_ORB_FEATURES << std::endl;
    std::cout << "  Max input size: " << MAX_INPUT_SIZE << "px" << std::endl;
    std::cout << "  Max panorama: " << MAX_PANORAMA_WIDTH << "x" << MAX_PANORAMA_HEIGHT << std::endl;
    std::cout << "  Compression threshold: " << std::fixed << std::setprecision(0) << (COMPRESSION_THRESHOLD * 100) << "% of max size (dynamic)" << std::endl;
    std::cout << "  Max images per sequence: ALL (with dynamic compression)" << std::endl;
    std::cout << std::endl;
    
    // default run
    if (argc == 1) {
        if (RUN_PAIR_EXPERIMENTS) {
            std::cout << "=== PAIR STITCHING EXPERIMENTS (2nd + 4th images) ===" << std::endl;
            std::cout << "Quality Control Status: " << (DISABLE_QUALITY_CONTROL ? "DISABLED" : "ENABLED") << std::endl << std::endl;
            
            std::vector<std::string> datasets = {"data/indoor", "data/outdoor", "data/wall"};
            
            for (const auto& datasetPath : datasets) {
                if (!std::filesystem::exists(datasetPath)) {
                    std::cout << "Warning: Dataset path does not exist: " << datasetPath << std::endl;
                    continue;
                }
                
                std::string datasetName = getDatasetName(datasetPath);
                std::vector<std::string> imageFiles = getImageFiles(datasetPath);
                
                if (imageFiles.size() < 4) {
                    std::cout << "Skipping " << datasetName << ": Need at least 4 images for pair experiments" << std::endl;
                    continue;
                }
                
                std::cout << "=== PAIR EXPERIMENTS - DATASET: " << datasetName << " ===" << std::endl;
                runPairStitchingExperiments(imageFiles, datasetName);
            }
            
            std::cout << "\n=== PAIR STITCHING EXPERIMENTS COMPLETED ===" << std::endl;
            std::cout << "Results saved in pair_stitching/ folders" << std::endl << std::endl;
        }
        
        if (RUN_FULL_PANORAMA_EXPERIMENTS) {
            std::cout << "=== FULL PANORAMA PROCESSING ===" << std::endl;
            std::cout << "Quality Control: ENABLED" << std::endl;
            std::cout << "Processing all images in each dataset" << std::endl << std::endl;
            
            std::vector<std::string> datasets = {"data/indoor", "data/outdoor", "data/wall"};
            
            for (const auto& datasetPath : datasets) {
                if (!std::filesystem::exists(datasetPath)) {
                    std::cout << "Warning: Dataset path does not exist: " << datasetPath << std::endl;
                    continue;
                }
                
                std::string datasetName = getDatasetName(datasetPath);
                std::cout << "\n=== PROCESSING DATASET: " << datasetName << " ===" << std::endl;
                
                std::filesystem::create_directories("output/" + datasetName);
                
                std::vector<std::string> imageFiles = getImageFiles(datasetPath);
                
                if (imageFiles.empty()) {
                    std::cout << "No images found in " << datasetPath << std::endl;
                    continue;
                }
                
                std::cout << "Found " << imageFiles.size() << " images:" << std::endl;
                for (const auto& img : imageFiles) {
                    std::cout << "  " << std::filesystem::path(img).filename().string() << std::endl;
                }
                
                if (imageFiles.size() < 2) {
                    std::cout << "Need at least 2 images for stitching!" << std::endl;
                    continue;
                }
                
                cv::Mat fullPanorama = stitchMultipleImagesOptimized(imageFiles, datasetName);
                
                if (!fullPanorama.empty()) {
                    std::string panoramaPath = "output/" + datasetName + "/" + datasetName + "_panorama.jpg";
                    if (cv::imwrite(panoramaPath, fullPanorama)) {
                        std::cout << "\nDataset " << datasetName << " completed successfully!" << std::endl;
                        std::cout << "Final panorama saved to: " << panoramaPath << std::endl;
                        std::cout << "Final panorama size: " << fullPanorama.cols << "x" << fullPanorama.rows << std::endl;
                    } else {
                        std::cout << "Failed to save panorama for " << datasetName << std::endl;
                    }
                    
                    if (RUN_EXPERIMENTS && imageFiles.size() >= 2) {
                        std::cout << "\nRunning experimental analysis on first two images..." << std::endl;
                        cv::Mat img1 = cv::imread(imageFiles[0]);
                        cv::Mat img2 = cv::imread(imageFiles[1]);
                        
                        if (!img1.empty() && !img2.empty()) {
                            std::string expOutputDir = "output/" + datasetName + "/experimental_analysis";
                            std::string imagePairName = datasetName + "_pair_1_2";
                            runComprehensiveExperiments(img1, img2, expOutputDir, imagePairName);
                        }
                    }
                } else {
                    std::cout << "Failed to create panorama for " << datasetName << std::endl;
                }
                
                fullPanorama.release();
            }
            
            std::cout << "\n=== ALL FULL PANORAMA PROCESSING COMPLETED ===" << std::endl;
            std::cout << "Results saved in respective output folders:" << std::endl;
            std::cout << "  output/indoor/" << std::endl;
            std::cout << "  output/outdoor/" << std::endl;
            std::cout << "  output/wall/" << std::endl;
        }
        
        return 0;
        
    } else if (argc == 2) {
        std::string datasetPath = argv[1];
        if (!std::filesystem::is_directory(datasetPath)) {
            std::cout << "Error: " << datasetPath << " is not a valid directory!" << std::endl;
            return -1;
        }
        
        std::string datasetName = getDatasetName(datasetPath);
        std::cout << "=== PROCESSING DATASET: " << datasetName << " ===" << std::endl;
        
        std::filesystem::create_directories("output/" + datasetName);
        
        std::vector<std::string> imageFiles = getImageFiles(datasetPath);
        
        if (imageFiles.empty()) {
            std::cout << "No -min.jpg images found in " << datasetPath << std::endl;
            return -1;
        }
        
        cv::Mat fullPanorama = stitchMultipleImagesOptimized(imageFiles, datasetName);
        
        if (!fullPanorama.empty()) {
            std::string panoramaPath = "output/" + datasetName + "/" + datasetName + "_panorama.jpg";
            cv::imwrite(panoramaPath, fullPanorama);
            std::cout << "Final panorama saved to: " << panoramaPath << std::endl;
            
            cv::namedWindow("Panorama Result", cv::WINDOW_AUTOSIZE);
            cv::imshow("Panorama Result", fullPanorama);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        
    // two image mode
    } else if (argc == 3) {
        cv::Mat img1 = cv::imread(argv[1]);
        cv::Mat img2 = cv::imread(argv[2]);
        
        if (img1.empty() || img2.empty()) {
            std::cout << "Error: Could not load one or both input images!" << std::endl;
            return -1;
        }
        
        std::cout << "Processing two images:" << std::endl;
        std::cout << "Image 1: " << argv[1] << " (" << img1.cols << "x" << img1.rows << ")" << std::endl;
        std::cout << "Image 2: " << argv[2] << " (" << img2.cols << "x" << img2.rows << ")" << std::endl;
        
        cv::Mat result = stitchTwoImagesOptimized(img1, img2);
        
        if (!result.empty()) {
            std::filesystem::create_directories("output");
            cv::imwrite("output/panorama_result.jpg", result);
            std::cout << "Result saved to: output/panorama_result.jpg" << std::endl;
            
            if (RUN_EXPERIMENTS) {
                std::cout << "\nRunning experimental analysis..." << std::endl;
                std::string expOutputDir = "output/experimental_analysis";
                std::string imagePairName = "two_image_analysis";
                runComprehensiveExperiments(img1, img2, expOutputDir, imagePairName);
            }
            
            cv::namedWindow("Panorama Result", cv::WINDOW_AUTOSIZE);
            cv::imshow("Panorama Result", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        
    } else {
        std::cout << "Usage:" << std::endl;
        std::cout << "  Process all datasets: " << argv[0] << std::endl;
        std::cout << "  Process one dataset: " << argv[0] << " <dataset_folder>" << std::endl;
        std::cout << "  Process two images: " << argv[0] << " <image1> <image2>" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << std::endl;
        std::cout << "  " << argv[0] << " data/indoor" << std::endl;
        std::cout << "  " << argv[0] << " data/indoor/indoor1-min.jpg data/indoor/indoor2-min.jpg" << std::endl;
        return -1;
    }
    
    return 0;
}