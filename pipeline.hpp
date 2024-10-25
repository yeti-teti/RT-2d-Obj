/*
Saugat Malla
Project 3
*/

#ifndef PIPELINE_H
#define PIPELINE_H

float SSD(const std::vector<float>& feature1, const std::vector<float>& feature2);

float scaledEuclideanDistance(const std::vector<float>& feature1, const std::vector<float>& feature2);

cv::Mat otsuThreshold(cv::Mat& src);

std::vector<float> computeFeatures(cv::Mat& region_mask, cv::Mat& frame);

void updateConfusionMatrix(const std::string& trueLabel, const std::string& predictedLabel, std::map<std::string, std::map<std::string, int>>& confusionMatrix);

std::string classifyWithKNN(const std::vector<float>& featureVector, const std::vector<std::vector<float>>& data, int k, const std::vector<char *> filenames);

std::vector<float> getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug=0 );

#endif