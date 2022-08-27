//
//  StartCamera.h
//  testCV
//
//  Created by Mason Kirby on 8/27/22.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


#ifndef StartCamera_h
#define StartCamera_h
#define startCamera _declspec(dllexport)

#endif /* StartCamera_h */

extern "C" {

startCamera int TurnCameraOn(){
// insert code here...
Mat frame;
   //--- INITIALIZE VIDEOCAPTURE
   VideoCapture cap;
   // open the default camera using default API
   // cap.open(0);
   // OR advance usage: select any API backend
   int deviceID = 0;             // 0 = open default camera
   int apiID = cv::CAP_ANY;      // 0 = autodetect default API
   // open selected camera using selected API
   cap.open(deviceID, apiID);
   // check if we succeeded
   if (!cap.isOpened()) {
       cerr << "ERROR! Unable to open camera\n";
       return -1;
   }
   //--- GRAB AND WRITE LOOP
   cout << "Start grabbing" << endl
       << "Press any key to terminate" << endl;
   for (;;)
   {
       // wait for a new frame from camera and store it into 'frame'
       cap.read(frame);
       // check if we succeeded
       if (frame.empty()) {
           cerr << "ERROR! blank frame grabbed\n";
           break;
       }
       // show live and wait for a key with timeout long enough to show images
       imshow("Live", frame);
       if (waitKey(5) >= 0)
           break;
   }
    return 0;
}
}

