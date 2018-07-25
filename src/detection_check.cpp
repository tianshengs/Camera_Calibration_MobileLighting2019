/*
Tommaso Monaco 2018
Middlebury College undergraduate summer research with Daniel Scharstein

This program is adapted from the OpenCV3's extra modules (AruCo), and Kyle Meredith's 2017 work (Middlebury College).
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
    may be used to endorse or promote products derived from this software
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

using namespace cv;
using namespace aruco;
using namespace std;


vector<  int  > returnVector;

//global variables for AruCo calibration 
vector< vector< Point2f > > allCornersConcatenated1_;
vector< int > allIdsConcatenated1_;
vector< int > markerCounterPerFrame1_;
vector< vector< Point2f > > allCornersConcatenated2_; // and for stereo AruCo calibration
vector< int > allIdsConcatenated2_;
vector< int > markerCounterPerFrame2_;

//struct to store parameters for intrinsic calibration
struct intrinsicCalibration_ {
  Mat cameraMatrix, distCoeffs;   //intrinsic camera matrices
  vector<Mat> rvecs, tvecs;       //extrinsic rotation and translation vectors for each image
  vector<vector<Point2f> > imagePoints;   //corner points on 2d image
  vector<vector<Point3f> > objectPoints;  //corresponding 3d object points
  vector<float> reprojErrs;   //vector of reprojection errors for each pixel
  double totalAvgErr = 0;     //average error across every pixel
  vector< vector< vector< Point2f > > > allCorners;
  vector< vector< int > > allIds;
  
};

//struct to store parameters for stereo calibration
struct stereoCalibration_ {
    Mat R, T, E, F;         //Extrinsic matrices (rotation, translation, essential, fundamental)
    Mat R1, R2, P1, P2, Q;  //Rectification parameters
                            //(rectification transformations, projection matrices, disparity-to-depth mapping matrix)
    Rect validRoi[2];       //Rectangle within the rectified image that contains all valid points
};


class ChessBoard_ : public aruco::GridBoard  {

public:
  void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);
  
  static Ptr<ChessBoard_> create_(int markersX, int markersY, float markerLength,
                                float markerSeparation,
                                const Ptr<aruco::Dictionary> &dictionary,
                                int firstMarker = 0);

  Ptr<aruco::Dictionary> localDictionary;
  
  std::vector< std::vector< Point3f >> obj_points_vector;
  
  std::vector< int > ids_vector;

  vector< int > ids;

  vector< vector< Point2f > > corners, rejected;


private:
  // number of markers in X and Y directions
  int __markersX, __markersY;
  
  // marker side lenght (normally in meters)
  float __markerLength;
  
  // separation between markers in the grid
  float __markerSeparation;
  
};


class Settings_
{
public:
    Settings_() : goodInput(false) {}
    enum Pattern { CHESSBOARD, ARUCO_SINGLE, NOT_EXISTING };
    enum Mode { INTRINSIC, STEREO,  INVALID };

    //Writes settings serialization to a file. Uncomment the other write() function
    //outside the settings class to use this functionality
    void write(FileStorage& fs) const
    {
        fs << "{" << "Mode" << modeInput
                  << "Calibration_Pattern" <<  patternInput

                  << "DetectedImages_Path" <<  detectedPath         
           << "}";
    }
    void read(const FileNode& node)             //Reads settings serialization
    {
        node["Mode"] >> modeInput;
        node["Calibration_Pattern"] >> patternInput;
        
        node["Num_MarkersX"] >> markersX;
        node["Num_MarkersY"] >> markersY;
        node["Marker_Length"] >> markerLength;
        node["Dictionary"] >> dictionary;
        node["First_Marker"] >> type;
        node["Num_of_Boards"] >> numberOfBoards; 
        
        node["DetectedImages_Path"] >> detectedPath;

       
        interprate();
    }
    void interprate()       //Interprets the settings and checks for valid input
    {
        goodInput = true;

        mode = INVALID;
        if (!modeInput.compare("INTRINSIC")) mode = INTRINSIC;
        if (!modeInput.compare("STEREO")) mode = STEREO;
       
        if (mode == INVALID)
            {
                cout << "Invalid calibration mode: " << modeInput << endl;
                goodInput = false;
            }

        calibrationPattern = NOT_EXISTING;
        if (!patternInput.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
        if (!patternInput.compare("ARUCO_SINGLE")) calibrationPattern = ARUCO_SINGLE;
        if (calibrationPattern == NOT_EXISTING)
            {
                cout << "Invalid calibration pattern: " << patternInput << endl;
                goodInput = false;
               
            }

        
	if (calibrationPattern ==  ARUCO_SINGLE){
	  if ((int) markersX.size() != numberOfBoards){
	    cout << "Invalid number of entries for Num_MarkersX" << endl;
	    goodInput = false;
	  }
	  else {
	    for (auto it=markersX.begin(); it!= markersX.end(); it++)
	      if (*it < 2){
		cout << "Invalid number of markers along x-axis: " <<  *it  << endl;
		goodInput = false;
	      }
	  }
	  
	
	  if ((int) markersY.size() != numberOfBoards){
	    cout << "Invalid number of entries for Num_MarkersY" << endl;
	    goodInput = false;
	  }
	  else {
	    for (auto it=markersY.begin(); it != markersY.end(); it++)
	      if (*it < 2){
		cout << "Invalid number of markers along Y-axis: " <<  *it << endl;
		goodInput = false;
	      }
	  }
        
          if ((int) markerLength.size() != numberOfBoards){
            cout << "Invalid number of entries for Marker_Length " <<  endl;
            goodInput = false;
          }
          else {
            for (auto it=markerLength.begin(); it != markerLength.end(); it++)
              if(*it < 1){
                cout << "Invalid marker length: " << endl;
                goodInput = false;
              }
          }


	  if (dictionary < 0 || dictionary > 16 ) {
	    cout << "Invalid dictionary:" <<  dictionary
		 << ". Check AruCo documentation if unsure. Fall back to default value: 11" << endl;
	    dictionary = 11;
	  }

          
          if ((int) type.size() != numberOfBoards){
            cout << "Invalid number of entries for First_Marker" <<  endl;
            goodInput = false;
          }
          for (auto it= type.begin(); it != type.end(); it++){
            if (*it < 0 || *it > 1000) {
              cout << "Invalid first marker id: " <<  *it << ". Fall back to default: 0"  << endl;
              *it = 0;
            }
          }
	    
          if (numberOfBoards < 1){
            cout << "Invalid number of boards used: " << numberOfBoards << ". Fall back to default: 1" << endl;
            numberOfBoards= 1;
          }
        }  
    }
              

  // Sets up the next image for pattern detection
  Mat imageSetup(int imageIndex)
  {
    Mat img;
    if( imageIndex < (int)imageList.size() )
      img = imread(imageList[imageIndex], CV_LOAD_IMAGE_COLOR);
    
    //If the image is too big, resize it. This makes it more visible and
        // prevents errors with ArUco detection.  
    if (img.cols>1080) resize(img, img, Size(), 0.5, 0.5);
    
    
    return img;
  }
  
    /*
    // Reads the image list from a file
    bool readImageList( const string& filename )
    {
        imageList.clear();
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened() )
            return false;
        FileNode n = fs.getFirstTopLevelNode();
        if( n.type() != FileNode::SEQ )
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for( ; it != it_end; ++it )
            imageList.push_back((string)*it);
        return true;
    }
    */
    


public:
//--------------------------Calibration configuration-------------------------//
    // Program modes:
    //    INTRINSIC  — calculates intrinsics parameters and  undistorts images
    //    STEREO     — calculates extrinsic stereo paramaters and rectifies images
 
  Mode mode;
  Pattern calibrationPattern;   // Three supported calibration patterns: CHESSBOARD, ARUCO_SINGLE, ARUCO_BOX
  
  Size boardSize;     // Size of chessboard (number of inner corners per chessboard row and column)
  float squareSize;   // The size of a square in some user defined metric system (pixel, millimeter, etc.)

//--------------------------AruCo configuration-------------------------//
  
     
  vector<int>  markersX;       // Number of AruCo Markers in first row
  vector<int>  markersY;       // Number of AruCo Markers in first column 
  vector<float> markerLength;  // The length of the markers in pixels
  int dictionary;              // The number of the AruCo dictionary used to draw the markers
  vector< int > type;          // The id of the first marker of the board
  int numberOfBoards;          // Number of boards in the scene. Default:1
  
//-----------------------------Input settings---------------------------------//
  vector<string> imageList;   // Image list to run calibration
  string imageListFilename;   // Input filename for image list
 
  //Intrinsic input can be used as an initial estimate for intrinsic calibration,
  //as fixed intrinsics for stereo calibration.
  //Leave filename at "0" to calculate new intrinsics
  intrinsicCalibration_ intrinsicInput; // Struct to store inputted intrinsics
  string intrinsicInputFilename;       // Intrinsic input filename
  bool useIntrinsicInput;              // Boolean to simplify program

//-----------------------------Output settings--------------------------------//
    string intrinsicOutput;    // File to write results of intrinsic calibration
    string extrinsicOutput;    // File to write results of stereo calibration

    // LEAVE THESE SETTINGS AT "0" TO NOT SAVE IMAGES
    string undistortedPath;    // Path at which to save undistorted images
    string rectifiedPath;      // Path at which to save rectified images
    string detectedPath;       // Path at which to save images with detected patterns

//-----------------------Intrinsic Calibration settings-----------------------//
    // It is recommended to fix distortion coefficients 3-5 ("00111"). Only 1-2 are needed
    // in most cases, and 3 produces significant distortion in stereo rectification
    string fixDistCoeffs;         // A string of five digits (0 or 1) that
                                  //   control which distortion coefficients will be fixed (1 = fixed)
    float aspectRatio;            // The aspect ratio. If it is non zero, it will be fixed in calibration
    bool assumeZeroTangentDist;   // Assume zero tangential distortion
    bool fixPrincipalPoint;       // Fix the principal point at the center
    int flag;                     // Flag to modify calibration

//--------------------------------UI settings---------------------------------//
    bool showUndistorted;   // Show undistorted images after intrinsic calibration
    bool showRectified;     // Show rectified images after stereo calibration
    bool wait;              // Wait until a key is pressed to show the next detected image

//-----------------------------Program variables------------------------------//
    int nImages;        // Number of images in the image list
    Size imageSize;     // Size of each image
    int nBoards;       // Number of marker maps read from config list

  
    bool goodInput;         //Tracks input validity
private:
    // Input variables only needed to set up settings
    string modeInput;
    string patternInput;
};    

static void read(const FileNode& node, Settings_ & x, const Settings_ & default_value = Settings_())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}


// Uncomment write() if you want to save your settings, using code like this:
        // FileStorage fs("settingsOutput.yml", FileStorage::WRITE);
        // fs << "Settings" << s;

// static void write(FileStorage& fs, const string&, const Settings& x)
// {
//     x.write(fs);
// }


Ptr<ChessBoard_> ChessBoard_::create_(int markersX, int markersY, float markerLength,
                                   float markerSeparation,
                                   const Ptr<aruco::Dictionary> &dictionary, int firstMarker) {

    Ptr<ChessBoard_> res = makePtr<ChessBoard_>();

    res->__markersX = markersX;
    res->__markersY = markersY;
    res->__markerLength = markerLength;
    res->__markerSeparation = markerSeparation;
    res->dictionary = dictionary;

    size_t totalMarkers = (size_t) ( markersX * markersY )+ ( (markersX-1) * (markersY-1) );
    res->ids.resize(totalMarkers);
    res->objPoints.reserve(totalMarkers);

    for(unsigned int i = 0; i < totalMarkers; i++) {
      res->ids[i] = i + firstMarker ;
    }

    // calculate Board objPoints
    float maxY = (float)markersY * markerLength + (markersY - 1) * markerSeparation;
    int numberSquaresY = markersY + (markersY -1);
    for(int y = 0; y < numberSquaresY; y++) {
      if (y%2 == 0){
        for(int x = 0; x < markersX; x++) {
            vector< Point3f > corners;
            corners.resize(4);
              corners[0] = Point3f(x * (markerLength + markerSeparation),
                                   maxY - y * (markerLength), 0);
              corners[1] = corners[0] + Point3f(markerLength, 0, 0);
              corners[2] = corners[0] + Point3f(markerLength, -markerLength, 0);
              corners[3] = corners[0] + Point3f(0, -markerLength, 0);
              res->objPoints.push_back(corners);
        }
      } else {
        for(int x = 0; x < markersX-1; x++) {
            vector< Point3f > corners;
            corners.resize(4);
            corners[0] = Point3f(((x+1)*markerSeparation) + (x*markerLength),
                                 maxY - y * (markerLength), 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + Point3f(0, -markerLength, 0);
            res->objPoints.push_back(corners);
            
        }
        
      }
     
    }
    
    res->localDictionary = dictionary;
    res->obj_points_vector = res->objPoints;
    res->ids_vector = res->ids;
    return res;
}



//----------------Error checking/Debugging helper functions-------------------//
// Checks if a path points to an existing directory
bool pathCheck_(const string& path)
{
    DIR* dir = opendir(path.c_str());

    if (dir)              // If the path is an actual directory
    {
        closedir(dir);
        return true;
    }
    else                  // Directory does not exist
        return false;
}

// Legibly prints the contents of a matrix
void printMat_(Mat m, const char *name)
{
    Size s = m.size();
    printf("%s: \t[", name);
    for (int i=0; i < s.height; i++)
    {
        for (int j=0; j < s.width; j++)
            printf("%.2f, ", m.at<double>(i,j));
    }
    cout << "]\n\n";
}

// Legibly prints the points of an intrinsic calibration struct
void printPoints_(const intrinsicCalibration_ inCal)
{
    for (auto v:inCal.objectPoints)
    {
        cout << "object " << v.size() << endl << "[";
        for (auto p:v)
             cout << " " << p << " ";
        cout << endl << endl;
    }
    /*
    for (auto v:inCal.imagePoints)
    {
        cout << "image " << v.size() << endl << "[";
        for (auto p:v)
             cout << " " << p << " ";
        cout << endl << endl;
    }
    */
}


//----------------------------------------------------------------------------

 
// Stereo calibration requires both images to have the same # of image and object points, but
// ArUco detections can include an arbitrary subset of all markers.
// This function limits the points lists to only those points shared between each image
void getSharedPoints_(intrinsicCalibration_ &inCal, intrinsicCalibration_ &inCal2)
{
    // pointers to make code more legible
    vector<Point3f> *oPoints, *oPoints2;
    vector<Point2f> *iPoints, *iPoints2;
    int shared;     //index of a shared object point
    bool paddingPoints = false;
    
    //for each objectPoints vector in overall objectPoints vector of vectors
    for (int i=0; i< (int)inCal.objectPoints.size(); i++)
    {
      std::map< string , int> countMap;
      vector<Point3f> sharedObjectPoints;
      vector<Point2f> sharedImagePoints, sharedImagePoints2; //shared image points for each inCal

        oPoints = &inCal.objectPoints.at(i);
        oPoints2 = &inCal2.objectPoints.at(i);
        iPoints  = &inCal.imagePoints.at(i);
        iPoints2 = &inCal2.imagePoints.at(i);

	
        if ((int)oPoints->size() >= (int)oPoints2->size()){
	  for (int j=0; j<(int)oPoints->size(); j++){
	      if (oPoints->at(0) == Point3f(-1,-1,0)) {		
		paddingPoints = true;
		break;
	      }
	      
	      for (shared=0; shared<(int)oPoints2->size(); shared++)
		if (oPoints->at(j) == oPoints2->at(shared)) break;
	      if (shared != (int)oPoints2->size())       //object point is shared
		{
		  stringstream temp;
		  temp << "(" << oPoints->at(j).x
		       << "," << oPoints->at(j).y
		       << "," << oPoints->at(j).z << ")";
		  auto result = countMap.insert(std::pair< string, int>(temp.str() , 1));
		  if (result.second == false)
		      result.first->second++;
		  if (result.second != 1)
		    continue;
		  sharedObjectPoints.push_back(oPoints->at(j));
		  sharedImagePoints.push_back(iPoints->at(j));
		  sharedImagePoints2.push_back(iPoints2->at(shared));
		}
	      paddingPoints = false;
	    }
	  }
      
	else {
	  for (int j=0; j<(int)oPoints2->size(); j++) {
	      if (oPoints2->at(0) == Point3f(-1,-1,0)) {		
		paddingPoints = true;
		break;
	      }
	      
	      for (shared=0; shared<(int)oPoints->size(); shared++)
		if (oPoints2->at(j) == oPoints->at(shared))
		  break;
	      if (shared != (int)oPoints->size())       //object point is shared
		{
		  stringstream temp;
		  temp << "(" << oPoints2->at(j).x
		       << "," << oPoints2->at(j).y
		       << "," << oPoints2->at(j).z << ")";
		  auto result = countMap.insert(std::pair< string, int>(temp.str() , 1));
		  if (result.second == false)
		      result.first->second++;
		  if (result.second != 1)
		    continue;
		  sharedObjectPoints.push_back(oPoints2->at(j));
		  sharedImagePoints2.push_back(iPoints2->at(j));
		  sharedImagePoints.push_back(iPoints->at(shared));
		}
	      paddingPoints = false;
	    }
	  }
	
	  
	
	if ((int) sharedObjectPoints.size() > 10){
	  
	  *oPoints = sharedObjectPoints;
	  *oPoints2 = sharedObjectPoints;
	  *iPoints = sharedImagePoints;
	  *iPoints2 = sharedImagePoints2;	
	}
	
	else if ((int) sharedObjectPoints.size() < 10 || paddingPoints) {
	  
	  inCal.objectPoints.erase(inCal.objectPoints.begin()+i);
	  inCal2.objectPoints.erase(inCal2.objectPoints.begin()+i);
	  inCal.imagePoints.erase(inCal.imagePoints.begin()+i);
	  inCal2.imagePoints.erase(inCal2.imagePoints.begin()+i);
	  
	  // temp
	  if (inCal.objectPoints.size() <= 0){
	    inCal.objectPoints[0].clear();
	    inCal2.objectPoints[0].clear();
	    break;
	  }
	  
	  
	  // decrement i because we removed one element
	  //  from the beginning of the vector, inCal.objectPoints.
	  i--; 
	}
    }
}

void getObjectAndImagePoints_( vector< vector< Point2f > >  detectedCorners, vector< int > detectedIds, vector< Point3f > &objPoints, vector< Point2f > &imgPoints, Ptr<ChessBoard_> &currentBoard) {

  std::map< string , int> countMap;
  size_t nDetectedMarkers = detectedIds.size();
  
  
  objPoints.reserve(nDetectedMarkers);
  imgPoints.reserve(nDetectedMarkers);

  // look for detected markers that belong to the board and get their information
  for(unsigned int i = 0; i < nDetectedMarkers; i++) {
    int currentId = detectedIds[i];
    for(unsigned int j = 0; j < currentBoard->ids_vector.size(); j++) {
      if(currentId == currentBoard->ids_vector[j]) {
	for(int p = 0; p < 4; p++) {
	  stringstream temp;
	  temp << "("
	       << currentBoard->obj_points_vector[j][p].x << ","
	       << currentBoard->obj_points_vector[j][p].y << ","
	       << currentBoard->obj_points_vector[j][p].z << ")";
	  auto result = countMap.insert(std::pair< string , int>(temp.str() , 1));
	  if (result.second == false)
	    result.first-> second++;
	  if (result.second !=1)
	    continue;
	  objPoints.push_back(currentBoard->obj_points_vector[j][p]);
	  imgPoints.push_back(detectedCorners[i][p]);
                    
	}
      }
    }
  }
}


void processPoints_(vector< vector< Point2f > > corners, vector<int> ids,
                        vector<int> counter,                      
                        vector< vector < Point2f >> &processedImagePoints,
                        vector< vector<Point3f>> &processedObjectPoints,  Ptr<ChessBoard_> &currentBoard) {

  // For each frame, get properly processed imagePoints and objectPoints
  //  for the calibrateCamera function
  size_t nFrames = counter.size();
  int markerCounter = 0;
  for(size_t frame = 0; frame < nFrames; frame++) {
    int nMarkersInThisFrame =  counter[frame];
    vector< vector< Point2f > > thisFrameCorners;
    vector< int > thisFrameIds;
    
    thisFrameCorners.reserve((size_t) nMarkersInThisFrame);
    thisFrameIds.reserve((size_t) nMarkersInThisFrame);
    
    for(int j = markerCounter; j < markerCounter + nMarkersInThisFrame; j++) {
      thisFrameCorners.push_back(corners[j]);
      thisFrameIds.push_back(ids[j]);
    }
    
    markerCounter += nMarkersInThisFrame;
    vector< Point2f > currentImgPoints;
    vector< Point3f > currentObjPoints;

    getObjectAndImagePoints_(thisFrameCorners, thisFrameIds, currentObjPoints,
                            currentImgPoints, currentBoard);
    if(currentImgPoints.size() > 0 && currentObjPoints.size() > 0) {
      processedImagePoints.push_back(currentImgPoints);
      processedObjectPoints.push_back(currentObjPoints);
    }

    else {
      for (int i=0; i < 4; i++){
	currentImgPoints.push_back(Point2f(-1,-1));
	currentObjPoints.push_back(Point3f(-1,-1,0));
      }
      processedImagePoints.push_back(currentImgPoints);
      processedObjectPoints.push_back(currentObjPoints);
    }
  }
}

void setUpAruco_( Settings_ s, intrinsicCalibration_ &inCal, intrinsicCalibration_ &inCal2, Ptr<ChessBoard_> &currentBoard, int n){

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
  
  processPoints_(allCornersConcatenated1_,
		allIdsConcatenated1_, markerCounterPerFrame1_,
		processedImagePoints1, processedObjectPoints1, currentBoard);
  
  inCal.objectPoints = processedObjectPoints1;
  inCal.imagePoints =processedImagePoints1;
  
  // If stereo mode, repeat the process for the second viewpoint
  if(s.mode == Settings_::STEREO){

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
    vector< vector<Point3f>> processedObjectPoints2;
    
    processPoints_(allCornersConcatenated2_,
		       allIdsConcatenated2_, markerCounterPerFrame2_,
		       processedImagePoints2, processedObjectPoints2, currentBoard);
    
    inCal2.objectPoints = processedObjectPoints2;
    inCal2.imagePoints =processedImagePoints2;

    
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
    
    cout << " # of objectPoints for image0"
	 << " and the " << s.markersX[n] << "x" << s.markersY[n] << " board"
	 << " with marker size " << s.markerLength[n]
	 << " is " 
	 << detected0 << endl;
    cout << " # of objectPoints for image1"
	 << " and the " << s.markersX[n] << "x" << s.markersY[n] << " board"
	 << " with marker size " << s.markerLength[n]
	 << " is " 
	 << detected1 << endl;

    returnVector.push_back(detected0);
    returnVector.push_back(detected1);
    
  }
}



void  arucoDetect_(Settings_ s, Mat &img, intrinsicCalibration_ &InCal, Ptr<ChessBoard_> currentBoard){


  Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

  detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
  //detectorParams-> doCornerRefinement = true; // do corner refinement in markers
  detectorParams-> cornerRefinementWinSize = 4;
  detectorParams->  minMarkerPerimeterRate = 0.01;
  detectorParams->  maxMarkerPerimeterRate = 4 ;
  detectorParams-> cornerRefinementMinAccuracy = 0.05;

  
  Mat imgCopy;
  
       
  // detect markers
  aruco::detectMarkers(img, currentBoard->localDictionary,
                       currentBoard->corners, currentBoard->ids,
                       detectorParams, currentBoard->rejected);

 
  
  if (currentBoard->ids.size() > 0){ 
    InCal.allCorners.push_back(currentBoard->corners); 
    InCal.allIds.push_back(currentBoard->ids);
    s.imageSize = img.size();
  }
  
  else if(currentBoard->ids.size() == 0 && s.mode == Settings_::STEREO) {

    vector < Point2f > temp;
    
    for (int i=0 ; i < 4; i++){
      temp.push_back(Point2f(-1,-1));
    }
    currentBoard->corners.push_back(temp);
    currentBoard->ids.push_back(-1);

    InCal.allCorners.push_back(currentBoard->corners);
    InCal.allIds.push_back(currentBoard->ids);
    
  }
  
}








//-----------------------------------------------------------------------------


// Main function. Detects patterns on images, runs calibration and saves results
vector<int> detectionCheck( char* settingsFile, char* filename0, char* filename1  )

{

  

  string inputSettingsFile = settingsFile;
 

  Mat img0;
  img0 = imread( filename0, CV_LOAD_IMAGE_COLOR );
  Mat img1;
  img1 = imread( filename1, CV_LOAD_IMAGE_COLOR );
  

  Settings_ s;
  FileStorage fs(inputSettingsFile, FileStorage::READ);   // Read the settings
  if (!fs.isOpened())
    {
      cout << "Could not open the settings file: \"" << inputSettingsFile << "\"" << endl;
      return returnVector;
    }
  fs["Settings"] >> s;
  fs.release();                                         // close Settings file
  
  if (!s.goodInput)
    {
      cout << "Invalid input detected. Application stopping. " << endl;
      return returnVector;
    }

  
  // struct to store calibration parameters
  intrinsicCalibration_ inCal, inCal2;
  intrinsicCalibration_ *currentInCal = &inCal;
  
  // size for stereo calibration
  int size = (s.mode == Settings_::STEREO) ? s.nImages/2 : s.nImages;
   
  
  char imgSave[1000];
  bool save = false;
  if(s.detectedPath != "0")
    {
      if( pathCheck_(s.detectedPath) )
	save = true;
      else
	printf("\nDetected images could not be saved. Invalid path: %s\n", s.detectedPath.c_str());
    }
  

    
/*-----------Calibration using AruCo patterns--------------*/ 
    if (s.calibrationPattern != Settings_::CHESSBOARD) {
    
      // all the AruCo structures are stored into lists (based on the # of boards in the scene)
      vector< Ptr<ChessBoard_> > boardsList;
      vector< vector < intrinsicCalibration_ > > inCalList;
      vector < intrinsicCalibration_ > tempList;
      Ptr<aruco::Dictionary> dictionary =
	aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(s.dictionary));     

 
     
      // initialize the AruCo structures
      for(int n = 0; n< s.numberOfBoards; n++){

	Ptr<ChessBoard_> board = ChessBoard_::create_(s.markersX[n], s.markersY[n],
						   s.markerLength[n], s.markerLength[n],
						   dictionary, s.type[n]);
	boardsList.push_back(board);
	
	// make intrinsicCalibration structs,
	//  two for each board.  
	intrinsicCalibration_ inCalAruCo;
	intrinsicCalibration_ inCalAruCo2;
	
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
      int value;

     
      for(int i = 0;;i++){
      	 
	  // Switches between intrinsic calibration structs for stereo mode
	  if (i%2 == 0) {
	    currentInCal = &inCalList[0][0];
	    value = 0;
	  } else if (s.mode == Settings_::STEREO){	    
	    currentInCal = &inCalList[ s.numberOfBoards-1][1];
	    value = 1;
	  }

	  Mat currentImg;
	  if (i == 0)
	    currentImg = img0;
	  else if ( i == 1)
	    currentImg = img1;
	  else
	    currentImg = Mat();

	  if(!currentImg.data) {

	    for(int n = 0; n< s.numberOfBoards; n++){
	      
	      setUpAruco_(s, inCalList[n][0], inCalList[n][1], boardsList[n], n);

	      getSharedPoints_(inCalList[n][0],  inCalList[n][1]);

	      cout << "Number of shared objectPoints for this board is " 
		   << inCalList[n][0].objectPoints[0].size() << endl;
	      if (inCalList[n][0].objectPoints[0].size() < 10)
		cout << "Not Good!" << endl;

	      returnVector.push_back((int)inCalList[n][0].objectPoints[0].size());
	      
	      
	    }
	    
	    break;
	    
	  }

	  s.imageSize = currentImg.size();
	  Mat imgCopy;
	    
	  for(int n = 0; n< s.numberOfBoards; n++){
	    arucoDetect_(s, currentImg, *currentInCal, boardsList[n]);
	    currentInCal = &inCalList[(i+1)% s.numberOfBoards][value];    
	    if(save) {
	      sprintf(imgSave, "%sdetected_%d.jpg", s.detectedPath.c_str(), i);
	      imwrite(imgSave, imgCopy);
	    }
	  }
	}
    }

    return returnVector;
}

