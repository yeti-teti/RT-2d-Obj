# RealTime-2d-ObjectDetection

Real Time 2-D Object Recognition

- Get real time feed from sensor
- Threshold the input video
    - Separates an object from the background
    - Preprocessing the video / image before thresholding 
        - Blur the image to make regions more uniform 
        - Making strongly colored pixels (pixels with high saturation value) darker and moving them further away from the white background which is unsaturated
    - Dynamically setting the threshold by analyzing the pixel values
    - K-Means algorithm is used to get the threshold
    - Display the thresholded video 
- Clean up the binary image
    - Using morphological filtering (erosion or dilation)
    - Checking if the image contains noise, holes or other issue
- Segment the image into regions
    - Connected component analysis on the thresholded and cleaned image to get regions
    - Displaying the found regions
    - Recognition of multiple objects simultaneously 
    - Finding the region map
- Compute features for each major region
    - Computes features for a specified region given a region map and region ID
- Collect Training Data
    - Collect featured vectors from objects for classification of unknown objects 
- Classify new images
    - System classifies new feature vector using the known object database and a distance metric
    - Labeled the unknown object to the closest matching feature vector in the objectDB (nearest-neighbor recognition)
- Performance evaluation of the system
    - Different orientations of different images
    - 5x5 confusion matrix of the results showing true labels versus classsified labels
- Using pre-trained DNN to create an embedding vector for the thresholded object and use the nearest-neighbor matching using distance metrics.



