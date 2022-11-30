//
//  StartCamera.h
//  testCV
//
//  Created by Mason Kirby on 8/27/22.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "detectObject.hpp"
#include <vector>
#include <unistd.h>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;



#ifndef StartCamera_h
#define StartCamera_h

CascadeClassifier faceDetect;
string name;
string fileName;
int fileNumber;
int numOfFiles = 0;

void detectBothEyes(){
    
    const string eyeCasacadeFilename = "/Users/masonkirby/Desktop/haarcascades/haarcascade_eye.xml";
    const float EYE_SX = 0.16f;
    const float EYE_SY = 0.26f;
    const float EYE_SW = 0.30f;
    const float EYE_SH = 0.28f;
    Mat face;
    
    VideoCapture cap(0);
    
    if(!faceDetect.load(eyeCasacadeFilename)){
        cout << "Error" << endl;
        return;
    }
    
    Mat frame;
    Mat res;
    Mat crop;
    cout << "Capturing your face 10 times, press c 10 times keeping you facec in front of the camera" << endl;
    
    char key;
    int i = 0;
    
    
    // Start of the loop to open the computer 
    for(;;){
        
        cap >> frame;
        imshow("Frame", frame);
     
     int leftX = cvRound(face.cols * EYE_SX);
     int topY = cvRound(face.rows * EYE_SY);
     int widthX = cvRound(face.cols * EYE_SW);
     int heightY = cvRound(face.rows * EYE_SH);
     int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner
     
     
     Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
     Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
     Rect leftEyeRect, rightEyeRect;
        detectLargestObject(topLeftOfFace, faceDetect, leftEyeRect, topLeftOfFace.cols);
        detectLargestObject(topRightOfFace, faceDetect, rightEyeRect, topLeftOfFace.cols);
        
        i++;
        
        if(i == 10){
            cout << "Face added!" << endl;
            break;
        }
        resize(frame, res, Size(128,128), 0, 0, INTER_LINEAR);
        stringstream ssfn;
        fileName = "/Users/masonkirby/Desktop/FACES";
        ssfn << fileName.c_str() << name << fileNumber << ".jpg";
        fileName = ssfn.str();
        imwrite(fileName, res);
        fileNumber++;
        waitKey(1000);
       
    }
    // For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
    
}

// Histogram Equalize seperately for the left and right sides of the face.
void equalizeLeftAndRightHalves(Mat &faceImg)
{
    // It is common that there is stronger light from one half of the face than the other. In that case,
    // if you simply did histogram equalization on the whole face then it would make one half dark and
    // one half bright. So we will do histogram equalization separately on each face half, so they will
    // both look similar on average. But this would cause a sharp edge in the middle of the face, because
    // the left half and right half would be suddenly different. So we also histogram equalize the whole
    // image, and in the middle part we blend the 3 images together for a smooth brightness transition.

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) First, equalize the whole face.
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize the left half and the right half of the face separately.
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Left 25%: just use the left face.
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Mid-left 25%: blend the left face & whole face.
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Mid-right 25%: blend the right face & whole face.
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%: just use the right face.
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// end x loop
    }//end y loop
}




void detectAndDisplay(Mat &frame){
    
    vector<Rect> faces;
    Mat grey_frame;
    Mat crop;
    Mat res;
    Mat grey;
    string text;
    stringstream sstm;
    
    cvtColor(frame, grey_frame, COLOR_BGR2GRAY);
    equalizeHist(grey_frame, grey_frame);
    
    faceDetect.detectMultiScale(grey_frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    
    Rect roi_b;
    Rect roi_c;
    
    size_t ic = 0;
    int ac = 0;
    
    size_t ib = 0;
    int ab = 0;
    

    for(int ic = 0; ic < faces.size(); ic++)
        {
              roi_c.x = faces[ic].x;
              roi_c.y = faces[ic].y;
              roi_c.width =  (faces[ic].width);
              roi_c.height = (faces[ic].height);
            
            ac = roi_c.width * roi_c.height;
            
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width =  (faces[ib].width);
            roi_b.height = (faces[ib].height);


//            rectangle(image, Point(x1, y1), Point(x2,y2), Scalar(50, 50, 255), 3);
//
//            putText(image, to_string(faces.size()), Point(10,40), FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,255), 1);
            
            crop = frame(roi_b);
            resize(crop, res, Size(128,128), 0, 0, INTER_LINEAR);
            cvtColor(crop, grey, COLOR_BGR2GRAY);
            stringstream ssfn;
            fileName = "/Users/masonkirby/Desktop/Faces";
            ssfn << fileName.c_str() << name << fileNumber << ".jpg";
            fileName = ssfn.str();
            imwrite(fileName, res);
            fileNumber++;
            
        }
}

void addFace(){
    
    cout << "enter your name" << endl;
    cin >> name;
    VideoCapture cap(0);
    
    if(!faceDetect.load("/Users/masonkirby/Desktop/face.xml")){
        cout << "Error" << endl;
        return;
    }
    
    Mat frame;
    cout << "Capturing your face 10 times, press c 10 times keeping you facec in front of the camera" << endl;
    
    char key;
    int i = 0;
    
    for(;;){
        
        cap >> frame;
        imshow("Frame", frame); 
        detectAndDisplay(frame);
        detectBothEyes();
//        cin >> key;
//        if(key == 'c'){
//            i++;
//        }
        i++;
        if(i == 10){
            cout << "Face added!" << endl;
            break;
        }
        
        waitKey(1000);
        
//        if(27 == char(c))
//        {
//            break;
//        }
       
    }
    
    return;
}

// dbread is Data base read. It is how the photos are going to populate the vectors
//
static void dbread(vector<Mat>& images, vector<int>& labels){
    vector<cv::String> fn;
    fileName = "/Users/masonkirby/Desktop/Faces2/";
    
    glob(fileName, fn, false);
    
    size_t count = fn.size();
    
    string itsname = "";
    char sep;
    
    for(size_t i = 0; i < count; i++){
        itsname ="";
        sep = '\\';
        size_t j = fn[i].rfind(sep, fn[i].length());
        if(j != string::npos)
        {
            itsname=(fn[i].substr(j+1, fn[i].length() - j-6));
        }
        images.push_back(imread(fn[i],0));
        labels.push_back(atoi(itsname.c_str()));
    }
    
}

// place the vector of photos inside of the machine learning alogrithm named EigenFaceRecognizer.train
void eigenFaceTrainer(){
    vector<Mat> images;
    vector<int> labels;
    dbread(images, labels);
    
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    
    model->train(images, labels);
    
    model->save("/Users/masonkirby/Desktop/Faces2/eigenFace.yml");
    
    cout << "Tarining Complete" << endl;
    waitKey(10000);
}

void faceRecognizer(){
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    
    model->read("/Users/masonkirby/Desktop/Faces2/eigenFace.yml");
    
    Mat testSample = imread("/Users/masonkirby/Desktop/Faces2/FACESMason0.jpg",0);
    
    int img_width = testSample.cols;
    int img_height = testSample.rows;
    
    string windows = "Capture - face detection";
    
    if(!faceDetect.load("/Users/masonkirby/Desktop/face.xml")){
        cout << "Error" << endl;
        return;
    }
    
    VideoCapture cap(0);
    
    if(!cap.isOpened()){
        cout << "exit" << endl;
        return;
    }
    namedWindow(windows, 1);
    long count = 0;
    string Pname = "";
    
    while(true)
    {
        vector<Rect> faces;
        Mat frame, crop;
        Mat grayScaleFrame;
        Mat Original;
        string name;
        
        cap >> frame;
        
        // count frames
        count = count + 1;
        
        if(!frame.empty())
        {
            //clone from original frame
            Original = frame.clone();
            
            // convert image to gray scale and equalize
            cvtColor(Original, grayScaleFrame, COLOR_BGR2GRAY);
            
            
            // detect face in gray images
            faceDetect.detectMultiScale(grayScaleFrame, faces, 1.1, 3, 0, cv::Size(70,70));
            
            //number of faces in gray image
            string fameset = to_string(count);
            string faceset = to_string(faces.size());
            
            int width, height = 0;
            
            for(int i = 0; i < faces.size(); i++)
            {
                Rect face_i = faces[i];
                
                Mat face = grayScaleFrame(face_i);
                
                Mat face_resized;
                
                resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
                
                int label = -1; double confidence = 0;
                
                model->predict(face_resized, label, confidence);
                
                cout << "Confidence: " << confidence << " Label " << label << endl;
                
                Pname = to_string(label);
                
                rectangle(Original, face_i, CV_RGB(0, 255, 0), 1);
                
                name = Pname;
                
                int pos_x = max(face_i.tl().x - 10, 0);
                int pos_y = max(face_i.tl().y - 10, 0);
                
                // name the person who is in the image
                putText(Original, name, Point(pos_x,pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0,255, 0), 1.0);
                
            }
            imshow(windows, Original);
            
        }
        if(waitKey(30) >= 0) break;
        
    }
    
}

#endif /* StartCamera_h */
