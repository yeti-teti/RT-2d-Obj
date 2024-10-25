/*
  Saugat Malla
  Project 3
*/

/*
    Run the file and then perform the key press based on requirements

    Key press:
    'q': Quit
    't': threshold the image
    'd': Closing operation
    'o': Opening operation
    's': Segmentation
    'r': Get region features (Collect training data)
    'n': Save features in the DB
    'c': Classify using scaled euclidean distance and nearest neighbors
    'k': Classify using KNN
    'b': Compute and save embeddings using DNN
    'm': Classify using DNN
*/

#include<iostream>           // Input-output stream operations
#include<fstream>            // File stream operations
#include<opencv2/opencv.hpp> // OpenCV library for computer vision tasks
#include<vector>             // Standard template library for vectors
#include<dirent.h>           // Directory handling
#include "csv_util.h"        // Custom header file for CSV file handling
#include<cstring>            // String manipulation operations
#include "pipeline.hpp"      // Custom header file for pipeline functions

#include <unordered_map>    // Hash table based associative containers
#include <map>              // Associative containers that store elements in a mapped fashion
#include <iomanip>          // Provides facilities to manipulate output formatting

using namespace std;

int main(){
    cv::VideoCapture *capdev;
    
    capdev = new cv::VideoCapture(0,cv::CAP_AVFOUNDATION); // Video capture object for accessing the camera
    if(!capdev -> isOpened()){
        cout<<"Unable to open video device"<<endl;
        return (-1);
    }

    // Get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT)
            );
    cout<<"Expected Size:"<<refS.width<<refS.height<<endl;

    // Identifies a window
    cv::namedWindow("Video", 1);
    cv::Mat frame;
    cv::Mat thresh,seg,dst;

    unordered_map<string, string> trueLabels;      // Map to store true labels
    unordered_map<string, string> predictedLabels; // Map to store predicted labels
    map<string, map<string, int>> confusionMatrix; // Confusion Matrix as a map of maps

    for(;;){
        *capdev >> frame; // Get a new frame from the camera, rear as a stream

        if(frame.empty()){
            cout<<"Frame is empty"<<endl;
            break;
        }

        imshow("Video", frame); // Display the captured frame

        // See if there is a waiting keystroke
        char key = cv::waitKey(10);

        if(key == 'q'){
            break;
        }
        if(key == 't'){ // Threshold the image

            thresh = otsuThreshold(frame);
            imshow("Otsu Thresholded Image", thresh);

        }
        if(key=='d'){ // Dialation

            thresh = otsuThreshold(frame);

            cv::morphologyEx(thresh, dst, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Point(3,3)));

            imshow("Dialation (Closing)", dst);

        }
        if(key=='o'){ // Erosion

            thresh = otsuThreshold(frame);

            cv::morphologyEx(thresh, dst, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Point(3,3)));

            imshow("Erosion (Opening)", dst);

        }
        if(key=='s'){ // Segment

            thresh = otsuThreshold(frame);

            cv::Mat morph;
            cv::morphologyEx(thresh, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(morph, labels, stats, centroids);

            cv::RNG rng(12345); // Random color generator

            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    frame.setTo(color, region_mask);
                }
            }

            imshow("Segmented Image", frame);

        }
        if (key == 'r') {  // Region features
            vector<float> featureVector;

            thresh = otsuThreshold(frame);

            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);

            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(), regionFeatures.begin(), regionFeatures.end());
                }
            }

            imshow("Features", frame);
        }
        if(key=='n'){ // Save features

            vector<float> featureVector;

            thresh = otsuThreshold(frame);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);

            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(), regionFeatures.begin(), regionFeatures.end());
                }
            }

            char csvfileName[256];
            strcpy(csvfileName, "featureVectors.csv");

            // Append features to CSV file
            char label[256];
            cout << "Enter the label name: ";
            cin >> label;
            append_image_data_csv(csvfileName, label, featureVector, 0);

            imshow("Features", frame);

        }
        if(key=='c'){ // Classify using nearest neighbor only
            char featureVectorFile[256];
            strcpy(featureVectorFile, "featureVectors.csv");

            // Reading feature vectors from file
            vector<char *> filenames; // Vector to store filenames
            vector<vector<float>> data; // Vector to store feature vectors
            if(read_image_data_csv(featureVectorFile, filenames, data)!=0){
                cout<<"Error: Unable to read feature vector file"<<endl;
                exit(-1);
            }

            // Compute feature vector for the current stream
            vector<float> featureVector;

            thresh = otsuThreshold(frame);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);

            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(), regionFeatures.begin(), regionFeatures.end());
                }
            }

            // Calculate scaled Euclidean distance and find nearest neighbor
            int nearestNeighborIndex = -1;
            float minDistance = numeric_limits<float>::max();

            for (size_t i = 0; i < data.size(); ++i) {
                float distance = scaledEuclideanDistance(featureVector, data[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestNeighborIndex = i;
                }
            }

            // Output label of the nearest neighbor object
            if (nearestNeighborIndex != -1) {
                string predictedLabel = filenames[nearestNeighborIndex];
                cout << "Nearest Neighbor Label: " << predictedLabel << " (Distance: " << minDistance << ")" << endl;

                // Display label on the output video stream
                cv::putText(frame, predictedLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                imshow("Predicted", frame);

                // Prompt user for feedback and update confusion matrix
                string trueLabel;
                cout << "Enter the predicted label: ";
                cin >> trueLabel;
                trueLabels["image_file_name"] = trueLabel;
                predictedLabels["image_file_name"] = predictedLabel;
                updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix);
            }
        }
        if(key=='k'){ // Classify using KNN

            char featureVectorFile[256];
            strcpy(featureVectorFile, "featureVectors.csv");

            // Reading feature vectors from file
            vector<char *> filenames; // Vector to store filenames
            vector<vector<float>> data; // Vector to store feature vectors

            if(read_image_data_csv(featureVectorFile, filenames, data)!=0){
                cout<<"Error: Unable to read feature vector file"<<endl;
                exit(-1);
            }

            // Compute feature vector for the current stream
            vector<float> featureVector;

            thresh = otsuThreshold(frame);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);

            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(), regionFeatures.begin(), regionFeatures.end());
                }
            }

            // Output label of the KNN
            int k=3;
            string predictedLabel = classifyWithKNN(featureVector, data, k, filenames);

            // Display label on the output video stream
            cv::putText(frame, predictedLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            imshow("Predicted", frame);

            cout << "KNN Label: " << predictedLabel << endl;

            // Prompt user for feedback and update confusion matrix
            string trueLabel;
            cout << "Enter the predicted label: ";
            cin >> trueLabel;
            trueLabels["image_file_name"] = trueLabel;
            predictedLabels["image_file_name"] = predictedLabel;
            updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix);

        }
        if(key=='b'){ // Save embeddings

            vector<float> featureVector;

            thresh = otsuThreshold(frame);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            // Embeddings
            char mod_filename[256];
            strcpy(mod_filename, "or2d-normmodel-007.onnx");

            // read the network
            cv::dnn::Net net = cv::dnn::readNet(mod_filename);
            printf("Network read successfully\n");
            cv::Rect bbox(0, 0, thresh.cols, thresh.rows);
            // get the embedding
            cv::Mat embedding;
            featureVector = getEmbedding(thresh, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging

            char csvfileName[256];
            strcpy(csvfileName, "featureVectorsDNN.csv");

            // Append features to CSV file
            char label[256];
            cout << "Enter the label name: ";
            cin >> label;
            append_image_data_csv(csvfileName, label, featureVector, 0);

            imshow("Features", frame);

        }
        if(key=='m'){ // Classify using DNN

            char featureVectorFile[256];
            strcpy(featureVectorFile, "featureVectorsDNN.csv");

            // Reading embeddings from file
            vector<char *> filenames; // Vector to store filenames
            vector<vector<float>> data; // Vector to store feature vectors
            if(read_image_data_csv(featureVectorFile, filenames, data)!=0){
                cout<<"Error: Unable to read feature vector file"<<endl;
                exit(-1);
            }

            // Compute embeddings for the current stream
            vector<float> featureVector;

            thresh = otsuThreshold(frame);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            // Embeddings
            char mod_filename[256];
            strcpy(mod_filename, "or2d-normmodel-007.onnx");

            // read the network
            cv::dnn::Net net = cv::dnn::readNet(mod_filename);
            printf("Network read successfully\n");
            cv::Rect bbox(0, 0, thresh.cols, thresh.rows);
            // get the embedding
            cv::Mat embedding;
            featureVector = getEmbedding(thresh, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging

            // Calculate SSD and find nearest neighbor
            int nearestNeighborIndex = -1;
            float minDistance = numeric_limits<float>::max();
            for (size_t i = 0; i < data.size(); ++i) {
                float distance = SSD(featureVector, data[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestNeighborIndex = i;
                }
            }

            // Output label of the nearest neighbor object
            if (nearestNeighborIndex != -1) {
                string predictedLabel = filenames[nearestNeighborIndex];
                cout << "Nearest Neighbor Label: " << predictedLabel << " (Distance: " << minDistance << ")" << endl;

                // Display label on the output video stream
                cv::putText(frame, predictedLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

                // Prompt user for feedback and update confusion matrix
                string trueLabel;
                cout << "Enter the predicted label: ";
                cin >> trueLabel;
                trueLabels["image_file_name"] = trueLabel;
                predictedLabels["image_file_name"] = predictedLabel;
                updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix);
            }
        }

        // Print confusion Matrix
        cout << "Confusion Matrix:" << endl;
        cout << setw(15) << "Predicted" << " | " << setw(15) << "True Label" << " | " << setw(10) << "Count" << endl;
        cout << "-------------------------------------------" << endl;

        for (auto it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it) {
            string trueLabel = it->first;
            for (auto jt = it->second.begin(); jt != it->second.end(); ++jt) {
                string predictedLabel = jt->first;
                int count = jt->second;
                cout << setw(15) << trueLabel << " | " << setw(15) << predictedLabel << " | " << setw(10) << count << endl;
            }
        }
    }

    delete capdev;

    return 0;
}
