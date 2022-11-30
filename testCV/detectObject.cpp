//
//  detectObject.cpp
//  testCV
//
//  Created by Mason Kirby on 11/29/22.
//

#include "detectObject.hpp"



// Search for objects such as faces in the image using the given parameters, storing the multiple cv::Rects into 'objects'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        cvtColor(img, gray, COLOR_BGRA2GRAY);
    }
    else {
        // Access the input image directly, since it is already grayscale.
        gray = img;
    }

    // Possibly shrink the image, to run much faster.
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Shrink the image while keeping the same aspect ratio.
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // Access the input image directly, since it is already small.
        inputImg = gray;
    }

    // Standardize the brightness and contrast to improve dark images.
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // Detect objects in the small grayscale image.
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // Enlarge the results if the image was temporarily shrunk before detection.
    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // Make sure the object is completely within the image, in case it was on a border.
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }

    // Return with the detected face rectangles stored in "objects".
}


void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    
    int flags = CASCADE_FIND_BIGGEST_OBJECT;
    
    Size minFeatureSize = Size(20,20);
    
    
    float searchScaleFactor = 1.1f;
    
    int minNeighbors = 4;
    
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if(objects.size() > 0){
        largestObject = (Rect)objects.at(0);
    }else{
        largestObject = Rect(-1,-1,-1,-1); 
    }
}
