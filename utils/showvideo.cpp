/* showvideo.cpp - display video in window
*
* Tiansheng Sun
*/
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/imgproc.hpp>


#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <string>

#include <vector>
#include <map>
#include <algorithm>
#include <functional>
//include "calibration.h"


using namespace cv;
using namespace std;
using namespace aruco;

const char *win = "video";

int cnt = 0;

//the function to draw detected ChArUco corners
void drawDetectedChArUco(Mat &image, vector<Point2f> corners, vector< int > ids = vector<int>()) {

    // calculate colors in Scalar(Blue, Green, Red)
    Scalar textColor, cornerColor, borderColor;
    borderColor = Scalar(51, 153, 255);
    cornerColor = Scalar(255, 0, 0);
    textColor = Scalar(255, 0, 0);

    int nMarkers = (int)corners.size();
    for(int i = 0; i < nMarkers; i++) {

        //draw a rectangle around the detected corner
        rectangle(image, corners[i] - Point2f(3,3), corners[i] + Point2f(3,3), cornerColor);

        // Draw the ChArUco corner Id if Ids given
        if(ids.size() > 0) {
            CV_Assert(corners.size() == ids.size());
            Point2f loc(0, 0);
            loc = corners[i] + Point2f(5,0); //set the location to draw text

            putText(image,std::to_string(ids[i]), loc, FONT_HERSHEY_COMPLEX_SMALL, 0.5, textColor, 1.9);
        }

    }

}

//the function to draw dectected Aruco patterns
void drawDetected(Mat &image, vector< vector< Point2f > > corners,  vector< int > ids = vector<int>()) {

    // calculate colors in Scalar(Blue, Green, Red)
    Scalar textColor, cornerColor, borderColor;
    borderColor = Scalar(51, 153, 255);
    cornerColor = Scalar(0, 50, 255);
    textColor = Scalar(0, 50, 255);

    int nMarkers = (int)corners.size();
    for(int i = 0; i < nMarkers; i++) {

        vector< Point2f > currentMarker = corners[i];
        CV_Assert(currentMarker.size() == 4);
        vector<Point> pt;
        for (int ao=0; ao< 4; ao++) {
          pt.push_back(currentMarker[ao]);
        }

        polylines(image, pt, false, borderColor);

        // Highlight the top right-hand corner of the current marker
        //  (within a second loop to avoid superposition of shapres)
        rectangle(image, currentMarker[0] - Point2f(3,3),
        currentMarker[0] + Point2f(3,3), cornerColor, 0.8, LINE_AA);

        // Draw the marker Id if Ids given
        if(ids.size() > 0) {
            Point2f loc(0, 0);
            loc = currentMarker[0] + Point2f(2, 15); //set the location to draw text

            putText(image,std::to_string(ids[i]), loc, FONT_HERSHEY_COMPLEX_SMALL, 0.6, textColor, 1.9);
            /*
            if (1) {
                int n = ids[i];
                if (n != 113 && n != 121)
                    continue;
                printf("img %4d  marker %d found\n", cnt, n);
                printf("  corner 0: %8.4f %8.4f\n", currentMarker[0].x, currentMarker[0].y);
                printf("  corner 1: %8.4f %8.4f\n", currentMarker[1].x, currentMarker[1].y);
                printf("  corner 2: %8.4f %8.4f\n", currentMarker[2].x, currentMarker[2].y);
                printf("  corner 3: %8.4f %8.4f\n", currentMarker[3].x, currentMarker[3].y);
            }
            */
        }

    }

}

//this is a helper function to generate ChArUco patterns
Ptr<aruco::CharucoBoard> generateCharucoBoard (int markersX, int markersY, float squareLength,
                                   float markerLength,
                                   const Ptr<aruco::Dictionary> &dictionary, int firstMarker) {

    //create a new Charuco board
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(markersX, markersY, squareLength, markerLength, dictionary);

    //calculate the total nubmer of markers
    size_t totalMarkers = (size_t) (markersX * markersY) + ((markersX - 1) * (markersY - 1));
    board->ids.resize(totalMarkers);

    //generate ids to be printed
    for(unsigned int i = 0; i < totalMarkers; i++) {
      board->ids[i] = i + firstMarker;
    }

    return board;

}

int main(int argc, char** argv)
{
    Mat frame;
    Mat frame_printed;
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners;
    Mat frameGray;
    vector< int > redefinedids;
    const char * inputSettingsFile;

    if (argc != 2) {
        cerr << "Usage: calibrateWithSettings [path to settings file]" << endl
             << "The settings folder contains several example files with "
                "descriptions of each parameter. Check the README for more detail." << endl;
        return -1;
    }
    else {
        inputSettingsFile = argv[1];
    }

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(11));
    int cam = 0; // default camera

    VideoCapture cap(cam);
    namedWindow(win, 1);
    if (!cap.isOpened()) {
        fprintf(stderr, "cannot open camera %d\n", cam);
        exit(1);
    }

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    //detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; //OpenCV 3.3 and above.
    detectorParams-> doCornerRefinement = true; // do corner refinement in markers // OpenCV 3.2 and below
    detectorParams-> cornerRefinementWinSize = 4;
    detectorParams->  minMarkerPerimeterRate = 0.01;
    detectorParams->  maxMarkerPerimeterRate = 4;
    detectorParams-> cornerRefinementMinAccuracy = 0.05;

    int i = 0;
    while (1) {
        vector< vector< Point2f > > rejected;
        vector<Point2f> corners;
        Mat cameraMatrix, distCoeffs;
        cap >> frame;
        cap >> frame_printed;

        cnt++;
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        Ptr<aruco::Dictionary> dictionary = getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(11));


        //define the variables used to create ChArUco pattern
        int squareX = 14;
        int squareY = 12;
        float squareLength = 120;
        float markerLength = 100;
        int firstmarker = 600;


        //define the variables used to create ChArUco pattern
        //int squareX2 = 14;
        //int squareY2 = 10;
        //float squareLength2 = 120;
        //float markerLength2 = 100;
        //int firstmarker2 = 600;

        //create and save a new Charuco board
        Ptr<aruco::CharucoBoard> board = generateCharucoBoard(squareX, squareY, squareLength, markerLength, dictionary, firstmarker);
        //Ptr<aruco::CharucoBoard> board2 = generateCharucoBoard(squareX2, squareY2, squareLength2, markerLength2, dictionary, firstmarker2);

        Scalar textColor, cornerColor, borderColor;
        borderColor = Scalar(51, 153, 255);
        cornerColor = Scalar(0, 50, 255);
        textColor = Scalar(0, 50, 255);

        //detect Aruco markers
        detectMarkers(frame, markerDictionary, markerCorners, markerIds, detectorParams, rejected);

        if(markerIds.size() > 0) {
            vector<Point2f> charucoCorners, charucoCorners2;
            vector<int> charucoIds, charucoIds2;

            drawDetected(frame, markerCorners, markerIds);
            interpolateCornersCharuco(markerCorners, markerIds, frame, board, charucoCorners, charucoIds);
            //interpolateCornersCharuco(markerCorners, markerIds, frame, board2, charucoCorners2, charucoIds2);

            // if at least one charuco corner detected
            if(charucoIds.size() > 0)
                drawDetectedChArUco(frame, charucoCorners, charucoIds);
            //if(charucoIds2.size() > 0)
            //    drawDetectedChArUco(frame, charucoCorners2, charucoIds2);

        }

       //uncomment to detect and draw chessboard corners
       /*
       //flag to be used to find chessboard corners
       int flag = CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS | CV_CALIB_CB_FAST_CHECK |
       CV_CALIB_CB_NORMALIZE_IMAGE;

       //find chessboard corners with specified size
       bool found = findChessboardCorners(frame, Size(7, 7), corners, flag);

       if (found)
       {
           //add some more precision
           cornerSubPix(frameGray, corners, Size(11,11), Size(-1,-1),
                        TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
       }

        drawChessboardCorners(frame, Size(7, 7), Mat(corners), found);
        */

        imshow(win, frame);

        int k = waitKey(30);
        //save images
        if (k == ' ') {
            printf("frame0%d.png saved\n ", i);
            imwrite(format("frame0%d.png", i), frame_printed);

            i = i + 1;
        }

        if (k == 'q'){
            break;/*
            FileStorage fs(inputSettingsFile, FileStorage:inputSettingsFil:WRITE);
            fs << "images" << "[";
            for (int j = 0; j < i; j++ ){
                string filename = format("frame0%d.png", j);
                fs << filename;
            }
            fs << "]";
            fs.release();
            //calibrateWithSettings(inputSettingsFile);
        */}
    }

    return 0;
}
