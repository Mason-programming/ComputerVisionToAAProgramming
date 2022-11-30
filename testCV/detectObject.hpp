//
//  detectObject.hpp
//  testCV
//
//  Created by Mason Kirby on 11/29/22.
//

#ifndef detectObject_hpp
#define detectObject_hpp
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;



void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaleWidth = 320);

#endif /* detectObject_hpp */
