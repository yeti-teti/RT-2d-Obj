/*
Saugat Malla
Project 3
*/

/*
    This file contains the functions required for the objectRecognition.cpp
*/

#include<iostream> // Standard input/output stream
#include<vector>    // Standard vector container
#include<opencv2/opencv.hpp> // OpenCV library
#include<algorithm> // Standard algorithms library

#include "csv_util.h"

using namespace std;

/**
 * @brief Calculates the Sum of Squared Differences (SSD) between two feature vectors.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return float SSD value
 */
float SSD(const vector<float>& feature1, const vector<float>& feature2){

    float ssd = 0, dx;

    for(size_t i=0;i<feature1.size();++i){
        dx = feature1[i] - feature2[i];
        ssd += dx*dx;
    }

    return ssd;
}

/**
 * @brief Computes the scaled Euclidean distance between two feature vectors.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return float Scaled Euclidean distance value
 */
float scaledEuclideanDistance(const vector<float>& feature1, const vector<float>& feature2) {
    if (feature1.size() != feature2.size()) {
        cerr << "Feature vectors must be of equal length" << endl;
        return numeric_limits<float>::max();
    }

    float sum = 0.0;
    for (size_t i = 0; i < feature1.size(); ++i) {
        float diff = (feature1[i] - feature2[i]);
        sum += (diff * diff);
    }

    float scaledDistance = sqrt(sum) / feature1.size(); // Divide by standard deviation or feature vector length
    return scaledDistance;
}

/**
 * @brief Computes Otsu's threshold manually to segment an input grayscale image into foreground and background.
 * 
 * @param src Input image
 * @return cv::Mat Thresholded image
 */
cv::Mat otsuThreshold(cv::Mat& src) {
    cv::Mat gray, blurred;

    // Convert source image to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    int hist[256] = {0};
    int total = blurred.rows * blurred.cols;

    // Calculate histogram
    for (int y = 0; y < blurred.rows; y++) {
        for (int x = 0; x < blurred.cols; x++) {
            hist[blurred.at<uchar>(y, x)]++;
        }
    }

    float sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += i * hist[i];
    }

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    int threshold = 0;

    for (int t = 0; t < 256; t++) {
        wB += hist[t]; // Weight Background
        if (wB == 0) continue;

        wF = total - wB; // Weight Foreground
        if (wF == 0) break;

        sumB += static_cast<float>(t * hist[t]);

        float mB = sumB / wB; // Mean Background
        float mF = (sum - sumB) / wF; // Mean Foreground

        // Calculate between-class variance
        float varBetween = static_cast<float>(wB) * static_cast<float>(wF) * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }

    // Create the destination matrix for the thresholded image
    cv::Mat dst;
    dst = cv::Mat::zeros(blurred.size(), CV_8U);

    // Apply thresholding based on the calculated threshold
    for (int y = 0; y < blurred.rows; y++) {
        for (int x = 0; x < blurred.cols; x++) {
            if (blurred.at<uchar>(y, x) > threshold) {
                dst.at<uchar>(y, x) = 255;
            } else {
                dst.at<uchar>(y, x) = 0;
            }
        }
    }

    return dst;
}

/**
 * @brief Computes features for a specified region of interest (ROI) in an image.
 * 
 * @param region_mask Binary mask representing the region of interest
 * @param frame Original image frame
 * @return vector<float> Feature vector
 */
vector<float> computeFeatures(cv::Mat& region_mask, cv::Mat& frame) {
    // Compute percent filled
    double filled_area = cv::countNonZero(region_mask);
    double total_area = region_mask.rows * region_mask.cols;
    double percent_filled = (filled_area / total_area) * 100.0;
    double min_region_area_threshold = 0.5;

    // Check if the region's area is below the threshold
    if (percent_filled < min_region_area_threshold) {
        // If below threshold, return empty feature vector
        return vector<float>();
    }

    // Compute bounding box
    cv::Rect bbox = cv::boundingRect(region_mask);

    // Calculate bounding box ratio
    double bounding_box_ratio = static_cast<double>(bbox.width) / bbox.height;

    // Draw bounding box
    cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

    // Compute moments
    cv::Moments moments = cv::moments(region_mask);

    // Compute additional features
    double area = moments.m00;
    double centroid_x = moments.m10 / area;
    double centroid_y = moments.m01 / area;
    double central_moments[3][3];
    double mu20 = moments.mu20 / area;
    double mu02 = moments.mu02 / area;
    double mu11 = moments.mu11 / area;
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    double nu20 = mu20 / (area * area);
    double nu02 = mu02 / (area * area);
    double nu11 = mu11 / (area * area);

    // Draw axis of least central moment
    cv::Point2f center(centroid_x, centroid_y);
    double half_length = 100.0; // Length of the axis
    cv::Point2f p1(center.x + half_length * cos(theta), center.y + half_length * sin(theta));
    cv::Point2f p2(center.x - half_length * cos(theta), center.y - half_length * sin(theta));
    cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);

    // Display percent filled on the frame
    stringstream ss;
    ss << "Filled: " << percent_filled << "%";
    cv::putText(frame, ss.str(), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    // Display centroid
    cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);

    // Store features in a vector
    vector<float> featureVector;
    featureVector.push_back(percent_filled);
    featureVector.push_back(bounding_box_ratio);
    featureVector.push_back(area);
    featureVector.push_back(centroid_x);
    featureVector.push_back(centroid_y);
    featureVector.push_back(mu20);
    featureVector.push_back(mu02);
    featureVector.push_back(mu11);
    featureVector.push_back(theta);
    featureVector.push_back(nu20);
    featureVector.push_back(nu02);
    featureVector.push_back(nu11);

    return featureVector;
}

/**
 * @brief Updates the confusion matrix based on the true and predicted labels of objects.
 * 
 * @param trueLabel True label of the object
 * @param predictedLabel Predicted label of the object
 * @param confusionMatrix Confusion matrix to be updated
 */
void updateConfusionMatrix(const string& trueLabel, const string& predictedLabel, map<string, map<string, int>>& confusionMatrix) {
    cout << "Predicted Label: " << predictedLabel << endl;
    cout << "Is this prediction correct? (y/n): ";
    char feedback;
    cin >> feedback;

    // Update confusion matrix based on user's feedback
    if (feedback == 'y' || feedback == 'Y') {
        // Prediction is correct
        confusionMatrix[trueLabel][predictedLabel]++;
    } else {
        // Prediction is incorrect, prompt for correct label
        string correctLabel;
        cout << "Enter the correct label: ";
        cin >> correctLabel;
        confusionMatrix[trueLabel][correctLabel]++;
    }
}

/**
 * @brief Finds the K nearest neighbors of a given feature vector in a dataset using the scaled Euclidean distance metric.
 * 
 * @param featureVector Feature vector for which nearest neighbors are to be found
 * @param data Dataset containing feature vectors
 * @param k Number of nearest neighbors to find
 * @param filenames Filenames corresponding to the data vectors
 * @return vector<pair<string, float>> Vector of filename-distance pairs for the K nearest neighbors
 */
vector<pair<string, float>> findKNearestNeighbors(const vector<float>& featureVector, const vector<vector<float>>& data, int k, const vector<char *> filenames) {
    vector<pair<string, float>> distances; // Vector to store filename-distance pairs

    // Compute distances from featureVector to all training data points
    for (size_t i = 0; i < data.size(); ++i) {
        float distance = scaledEuclideanDistance(featureVector, data[i]); // Assuming euclideanDistance function exists
        distances.emplace_back(filenames[i], distance); // Store filename-distance pair
    }

    // Sort distances vector based on distance
    sort(distances.begin(), distances.end(), [](const pair<string, float>& a, const pair<string, float>& b) {
        return a.second < b.second;
    });

    // Return the K nearest neighbors
    return vector<pair<string, float>>(distances.begin(), distances.begin() + k);
}

/**
 * @brief Classifies a given feature vector using the K Nearest Neighbors (KNN) algorithm with a specified value of K.
 * 
 * @param featureVector Feature vector to be classified
 * @param data Dataset containing feature vectors
 * @param k Number of nearest neighbors to consider
 * @param filenames Filenames corresponding to the data vectors
 * @return string Predicted label for the input feature vector
 */
string classifyWithKNN(const vector<float>& featureVector, const vector<vector<float>>& data, int k, const vector<char *> filenames) {
    vector<pair<string, float>> nearestNeighbors = findKNearestNeighbors(featureVector, data, k, filenames);

    // Count occurrences of each class label among the K nearest neighbors
    unordered_map<string, int> labelCounts;
    for (const auto& neighbor : nearestNeighbors) {
        labelCounts[neighbor.first]++;
    }

    // Find the class label with the highest count
    string predictedLabel;
    int maxCount = 0;
    for (const auto& pair : labelCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            predictedLabel = pair.first;
        }
    }

    return predictedLabel;
}

/**
 * @brief Extracts embeddings from a region of interest (ROI) in an image using a deep neural network (DNN) model and returns the feature vector.
 * 
 * @param src Source image
 * @param embedding Extracted embedding
 * @param bbox Bounding box representing the region of interest
 * @param net Pretrained neural network model
 * @param debug Flag indicating whether to display debug information
 * @return vector<float> Feature vector extracted from the ROI
 */
vector<float> getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 128;
  cv::Mat padImg;
  cv::Mat blob;
	
  cv::Mat roiImg = src( bbox );
  int top = bbox.height > 128 ? 10 : (128 - bbox.height)/2 + 10;
  int left = bbox.width > 128 ? 10 : (128 - bbox.width)/2 + 10;
  int bottom = top;
  int right = left;
	
  cv::copyMakeBorder( roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0  );
  cv::resize( padImg, padImg, cv::Size( 128, 128 ) );

  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) / 0.5, // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  128,   // subtract mean prior to scaling
			  false, // input is a single channel image
			  true,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!/fc1/Gemm" );

  vector<float> emb;
    for (int y = 0; y < embedding.rows; ++y) {
        for (int x = 0; x < embedding.cols; ++x) {
            emb.push_back(embedding.at<uchar>(y, x)); // Store pixel values in the feature vector
        }
    }


  if(debug) {
    cv::imshow( "pad image", padImg );
    std::cout << embedding << std::endl;
    cv::waitKey(0);
  }

  return emb;
}
