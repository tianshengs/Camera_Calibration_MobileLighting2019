/*
Tiansheng Sun
Middlebury College undergraduate summer research with Daniel Scharstein

This program is adapted from the OpenCV3's extra modules (AruCo), Kyle Meredith's 2017 work (Middlebury College), and
Tommaso Monaco's 2018 work.
This program inherits some aspects from calibrate_camera.cpp, an example calibration programs provided by opencv.

The program has three modes: intrinsic calibration, stereo calibration, and live
feed preview. It supports three patterns: chessboard, and AruCo single.

Read the README for more information and guidance.

----------------
From the AruCo Module (opencv_contrib), copyright notice:

By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this softwareTommaso Monaco 2018
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>


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
#include "calibration.h"
using namespace cv;
using namespace aruco;
using namespace std;

vector< vector< Point2f > > allCornersConcatenated1_;
vector< int > allIdsConcatenated1_;
vector< int > markerCounterPerFrame1_;
vector< vector< Point2f > > allCornersConcatenated2_; // and for stereo AruCo calibration
vector< int > allIdsConcatenated2_;
vector< int > markerCounterPerFrame2_;
vector<  int  > returnVector_;

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
        if(node.empty())
            x = default_value;
        else
            x.read(node);
};

// this funciont set up aruco for detection check
void setUpAruco_( Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2, Ptr<ChessBoard> &currentBoard, int n){

    // Clear these temporary storage vectors (decleared globally)
    allCornersConcatenated1_.clear();
    allIdsConcatenated1_.clear();
    markerCounterPerFrame1_.clear();

    markerCounterPerFrame1_.reserve(inCal.allCorners.size());
    for(unsigned int i = 0; i < inCal.allCorners.size(); i++) {
        markerCounterPerFrame1_.push_back((int)inCal.allCorners[i].size());
        for(unsigned int j = 0; j < inCal.allCorners[i].size(); j++) {
            allCornersConcatenated1_.push_back(inCal.allCorners[i][j]);
            allIdsConcatenated1_.push_back(inCal.allIds[i][j]);
        }
    }

    vector< vector < Point2f >>  processedImagePoints1;
    vector< vector<Point3f>> processedObjectPoints1;

    processPoints(s, allCornersConcatenated1_,
                   allIdsConcatenated1_, markerCounterPerFrame1_,
                   processedImagePoints1, processedObjectPoints1, currentBoard);

    inCal.objectPoints = processedObjectPoints1;
    inCal.imagePoints = processedImagePoints1;

    int total_corners = 4 * s.markersX[n] * s.markersY[n];

    if(s.mode == Settings::INTRINSIC) {

        int detected0;
        cout<<inCal.objectPoints.size();

        //if the image is not detected
        if (inCal.objectPoints.size() <= 0)
            detected0 = 0;
        else
            detected0 = inCal.objectPoints[0].size();

        float percentage = detected0 / (float)total_corners;

        //prints out the number of detected corners of this image
        cout << " Number of detected corners for this image"
        << " and the " << s.markersX[n] << "x" << s.markersY[n] << " board"
        << " with marker size " << s.markerLength[n]
        << " is "
        << detected0 << endl
        << "Percentage of corners detected:"
        << percentage << endl;

        returnVector_.push_back(detected0);

    }

    // If stereo mode, repeat the process for the second viewpoint
    else if(s.mode == Settings::STEREO){

        // Clear these temporary storage vectors (decleared globally)
        allCornersConcatenated2_.clear();
        allIdsConcatenated2_.clear();
        markerCounterPerFrame2_.clear();

        markerCounterPerFrame2_.reserve(inCal2.allCorners.size());
        for(unsigned int i = 0; i < inCal2.allCorners.size(); i++) {
            markerCounterPerFrame2_.push_back((int)inCal2.allCorners[i].size());
            for(unsigned int j = 0; j < inCal2.allCorners[i].size(); j++) {
                allCornersConcatenated2_.push_back(inCal2.allCorners[i][j]);
                allIdsConcatenated2_.push_back(inCal2.allIds[i][j]);
            }
        }

        vector< vector < Point2f >>  processedImagePoints2;
        vector< vector < Point3f >> processedObjectPoints2;

        processPoints(s, allCornersConcatenated2_,
                       allIdsConcatenated2_, markerCounterPerFrame2_,
                       processedImagePoints2, processedObjectPoints2, currentBoard);

        inCal2.objectPoints = processedObjectPoints2;
        inCal2.imagePoints = processedImagePoints2;


        int detected0;
        int detected1;

        if (processedObjectPoints1[0][0] == Point3f(-1,-1,0)
            && processedObjectPoints2[0][0] == Point3f(-1,-1,0))
        {
            detected0 = 0;
            detected1 = 0;
        }

        else
        {
            detected0 = inCal.objectPoints[0].size();
            detected1 = inCal2.objectPoints[0].size();
        }

        //print out the number of detected corners for each image
        cout << " Number of detected corners for image0"
        << " and the " << s.markersX[n] << "x" << s.markersY[n] << " board"
        << " with marker size " << s.markerLength[n]
        << " and a total of " << total_corners << " corners"
        << " is "
        << detected0 << endl;

        cout << " Number of detected corners for image1"
        << " and the " << s.markersX[n] << "x" << s.markersY[n] << " board"
        << " with marker size " << s.markerLength[n]
        << " and a total of " << total_corners << " corners"
        << " is "
        << detected1 << endl;

        returnVector_.push_back(detected0);
        returnVector_.push_back(detected1);

    }

}

// Main function. Detects patterns on images
vector<int> detectionCheck( char* settingsFile, char* filename0, char* filename1) {
    string inputSettingsFile = settingsFile;

    Mat img0;
    Mat img1;
    img0 = imread( filename0, CV_LOAD_IMAGE_COLOR );

    if (filename1)
        img1 = imread( filename1, CV_LOAD_IMAGE_COLOR );

    Settings s;
    FileStorage fs(inputSettingsFile, FileStorage::READ);   // Read the settings

    if (!fs.isOpened())
    {
        cout << "Could not open the settings file: \"" << inputSettingsFile << "\"" << endl;
        return returnVector_;
    }

    fs["Settings"] >> s;
    fs.release(); // close Settings file

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping." << endl;
        return returnVector_;
    }

    // struct to store calibration parameters
    intrinsicCalibration inCal, inCal2;
    intrinsicCalibration *currentInCal = &inCal;

    // size of vectors for stereo calibration
    int size = (s.mode == Settings::STEREO) ? s.nImages/2 : s.nImages;

    char imgSave[1000];
    bool save = false;
    if(s.detectedPath != "0")
    {
        if( pathCheck(s.detectedPath) )
            save = true;
        else
            printf("\nDetected images could not be saved. Invalid path: %s\n", s.detectedPath.c_str());
    }

    if(s.nImages < 0 ) {
        cout << "Failed to initialize number of image paths in stereo image list from: \"" << inputSettingsFile << "\"" << endl;
    }

    /*-----------Calibration using AruCo patterns--------------*/
    if (s.calibrationPattern != Settings::CHESSBOARD) {

        // all the AruCo structures are stored into lists (based on the # of boards in the scene)
        vector< Ptr<ChessBoard> > boardsList;
        vector< vector < intrinsicCalibration > > inCalList;
        vector < intrinsicCalibration > tempList;
        Ptr<aruco::Dictionary> dictionary =
            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(s.dictionary));

        // initialize the AruCo structures
        for(int n = 0; n < s.numberOfBoards; n++){

            Ptr<ChessBoard> board = ChessBoard::create(s.markersX[n], s.markersY[n],
                                                     s.markerLength[n], s.markerLength[n],
                                                     dictionary, s.type[n]);

            boardsList.push_back(board);

            // make intrinsicCalibration structs,
            //  two for each board.
            intrinsicCalibration inCalAruCo;
            intrinsicCalibration inCalAruCo2;

            // require correctly sized vectors
            inCalAruCo.imagePoints.resize(size);
            inCalAruCo.objectPoints.resize(size);
            inCalAruCo2.imagePoints.resize(size);
            inCalAruCo2.objectPoints.resize(size);

            // make a list of lists of the newly created intrinsicCalibration structs
            tempList.push_back(inCalAruCo);
            tempList.push_back(inCalAruCo2);
            inCalList.push_back(tempList);
        }

        // variable used to facilitate the alternation
        //  between intrinsic calibration structs for stereo mode.
        int value = 0;

        for(int i = 0;;i++) {

            // Switches between intrinsic calibration structs for stereo mode
            if (i % 2 == 0) {
                currentInCal = &inCalList[0][0];
                value = 0;
            } else if (s.mode == Settings::STEREO){
                currentInCal = &inCalList[ s.numberOfBoards-1][1];
                value = 1;
            }

            Mat currentImg;
            if (i == 0)
                currentImg = img0;
            else if ( i == 1 ){
                if ( filename1 )
                    currentImg = img1;
                else
                    currentImg = Mat();
            }
            else
                currentImg = Mat();

            // Print the number of detected shared object points, along with warnings if there are not enough
            if(!currentImg.data) {

                for(int n = 0; n < s.numberOfBoards; n++){
                    setUpAruco_(s, inCalList[n][0], inCalList[n][1], boardsList[n], n);

                    // if stereo, then print out the number of shared object Points
                    if (s.mode == Settings::STEREO){

                        getSharedPoints(inCalList[n][0],  inCalList[n][1]);

                        cout << "Number of shared objectPoints for this board is "
                            << inCalList[n][0].objectPoints[0].size() << endl;

                        if (inCalList[n][0].objectPoints[0].size() < 10)
                            cout << "Not enough object points! This set should be retaken." << endl;

                        returnVector_.push_back((int)inCalList[n][0].objectPoints[0].size());

                    }

                }

                break;
            }

            s.imageSize = currentImg.size();
            Mat imgCopy;

            for(int n = 0; n < s.numberOfBoards; n++){
                arucoDetect(s, currentImg, *currentInCal, boardsList[n]);
                currentInCal = &inCalList[(i+1)% s.numberOfBoards][value];
                if(save) {
                    sprintf(imgSave, "%sdetected_%d.jpg", s.detectedPath.c_str(), i);
                    imwrite(imgSave, imgCopy);
                }
            }
        }
    }

    return returnVector_;
}
