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

//struct to store parameters for intrinsic calibration
struct intrinsicCalibration {
  Mat cameraMatrix, distCoeffs;   //intrinsic camera matrices
  vector<Mat> rvecs, tvecs;       //extrinsic rotation and translation vectors for each image
  vector<vector<Point2f> > imagePoints;   //corner points on 2d image
  vector<vector<Point3f> > objectPoints;  //corresponding 3d object points
  vector<float> reprojErrs;   //vector of reprojection errors for each pixel
  double totalAvgErr = 0;     //average error across every pixel
};

//struct to store parameters for stereo calibration
struct stereoCalibration {
    Mat R, T, E, F;         //Extrinsic matrices (rotation, translation, essential, fundamental)
    Mat R1, R2, P1, P2, Q;  //Rectification parameters
                            //(rectification transformations, projection matrices, disparity-to-depth mapping matrix)
    Rect validRoi[2];       //Rectangle within the rectified image that contains all valid points
};


// Tommaso Monaco, Summer 2018
//------------------Intermediate Global Variables for AruCo Calibration----------------//
//   (after detection of the markers,
//   the detected object points need
//   to be available for processing, which
//   prepares the object points for calibration)
struct arucoCalibration {
    vector<vector<Point2f> > imgPoints;     // corner points on 2d image for each aruco board in the scene 
    vector<vector<Point3f> > objPoints;  // corresponding 3d object points prepared for calibration.
    
    vector< vector< vector< Point2f > > > allCorners; //all corners of detected markers
    vector< vector< int > > allIds; //corresponding ids. 

    vector< vector< Point2f > > allCornersConcatenated; //all detected corners concatenated per frame
    vector< int > allIdsConcatenated; //corresponding ids
    vector< int > markerCounterPerFrame; //number of images
};


class ChessBoard : public aruco::GridBoard  {

public:
  void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);
  
  static Ptr<ChessBoard> create(int markersX, int markersY, float markerLength,
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


class Settings {
public:
  
  enum Pattern { CHESSBOARD, ARUCO_SINGLE, NOT_EXISTING };
  enum Mode { INTRINSIC, STEREO,  INVALID };
  Settings() : goodInput(false),
	       mode(INVALID),
	       calibrationPattern(NOT_EXISTING) {}
  
  
  //Writes settings serialization to a file. Uncomment the other write() function
  //outside the settings class to use this functionality
  void write(  FileStorage& fs) {
    fs << "{"
	
       << "Mode" << modeInput
       << "Calibration_Pattern" <<  patternInput
	
       << "ChessboardSize_Width"  <<  boardSize.width
       << "ChessboardSize_Height" <<  boardSize.height
       << "SquareSize" << squareSize
	  
       << "ImageList_Filename" <<  imageListFilename	  
       << "IntrinsicInput_Filename" <<  intrinsicInputFilename
	  
       << "IntrinsicOutput_Filename" <<  intrinsicOutput
       << "ExtrinsicOutput_Filename" <<  extrinsicOutput
	
       << "UndistortedImages_Path" <<  undistortedPath
       << "RectifiedImages_Path" <<  rectifiedPath
       << "DetectedImages_Path" <<  detectedPath
	  
       << "Calibrate_FixDistCoeffs" << fixDistCoeffs
       << "Calibrate_FixAspectRatio" <<  aspectRatio
       << "Calibrate_AssumeZeroTangentialDistortion" <<  assumeZeroTangentDist
       << "Calibrate_FixPrincipalPointAtTheCenter" <<  fixPrincipalPoint
	  
       << "Alpha_parameter" << alpha
       << "Resizing_factor" << rf
       << "Cropping_After_Rectification" << crop
	  
       << "Show_UndistortedImages" <<  showUndistorted
       << "Show_RectifiedImages" <<  showRectified
       << "Wait_NextDetectedImage" << wait
	         
       << "}";
  }
  void read(const FileNode& node)             //Reads settings serialization
  {
    node["Mode"] >> modeInput;
    node["Calibration_Pattern"] >> patternInput;

    node["ChessboardSize_Width" ] >> boardSize.width;
    node["ChessboardSize_Height"] >> boardSize.height;
    node["SquareSize"]  >> squareSize;

    node["Num_MarkersX"] >> markersX;
    node["Num_MarkersY"] >> markersY;
    node["Marker_Length"] >> markerLength;
    node["Dictionary"] >> dictionary;
    node["First_Marker"] >> type;
    node["Num_of_Boards"] >> numberOfBoards; 
        
    node["ImageList_Filename"] >> imageListFilename;
    node["IntrinsicInput_Filename"] >> intrinsicInputFilename;

    node["IntrinsicOutput_Filename"] >> intrinsicOutput;
    node["ExtrinsicOutput_Filename"] >> extrinsicOutput;

    node["UndistortedImages_Path"] >> undistortedPath;
    node["RectifiedImages_Path"] >> rectifiedPath;
    node["DetectedImages_Path"] >> detectedPath;

    node["Calibrate_FixDistCoeffs"] >> fixDistCoeffs;
    node["Calibrate_FixAspectRatio"] >> aspectRatio;
    node["Calibrate_AssumeZeroTangentialDistortion"] >> assumeZeroTangentDist;
    node["Calibrate_FixPrincipalPointAtTheCenter"] >> fixPrincipalPoint;
	
    node["Alpha_parameter"] >> alpha;
    node["Resizing_factor"] >> rf;
    node["Cropping_After_Rectification"] >> crop;

    node["Show_UndistortedImages"] >> showUndistorted;
    node["Show_RectifiedImages"] >> showRectified;
    node["Wait_NextDetectedImage"] >> wait;
      
    interprate();
  }
  void interprate()       //Interprets the settings and checks for valid input
  {
    goodInput = true;

    if (!modeInput.compare("INTRINSIC")) mode = INTRINSIC;
    if (!modeInput.compare("STEREO")) mode = STEREO;
       
    if (mode == INVALID)
      {
	cout << "Invalid calibration mode: " << modeInput << endl;
	goodInput = false;
      }

    if (!patternInput.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
    if (!patternInput.compare("ARUCO_SINGLE")) calibrationPattern = ARUCO_SINGLE;
    if (calibrationPattern == NOT_EXISTING)
      {
	cout << "Invalid calibration pattern: " << patternInput << endl;
	goodInput = false;
               
      }
    if (calibrationPattern ==  CHESSBOARD){
      if (boardSize.width <= 0 || boardSize.height <= 0)
	    {
	      cout << "Invalid chessboard size: " << boardSize.width << " " << boardSize.height << endl;
	      goodInput = false;
	    }
	  if (squareSize <= 10e-6)
	    {
	      cout << "Invalid square size " << squareSize << endl;
	      goodInput = false;
	    }
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
        
        
      
        if (readImageList(imageListFilename))
        {
            nImages = (int)imageList.size();
            if (mode == STEREO)
                if (nImages % 2 != 0) {
                    cout << "Image list must have even # of elements for stereo calibration" << endl;
                    goodInput = false;
                }
        }
        else {
            cout << "Invalid image list: " << imageListFilename << endl;
            goodInput = false;
        }
        
        useIntrinsicInput = false;
        if (readIntrinsicInput(intrinsicInputFilename)) {
          useIntrinsicInput = true;
        }

        


        flag = 0;
        int digit, shift;
        // For each '1' digit in the fixDistCoeffs setting, add the fix flag
        for (int i=0; i<5; i++)
        {
            digit = fixDistCoeffs[i] - '0';   //gets first digit as int
            // The FIX_K[1-5] flags are separated by powers of 2, with a jump of 3 after K3
            if (i >= 3) shift = i + 3;
            else shift = i;
            if (digit)
                flag |= CV_CALIB_FIX_K1 << shift;
        }

        if(fixPrincipalPoint)       flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
        if(assumeZeroTangentDist)   flag |= CV_CALIB_ZERO_TANGENT_DIST;
        if(aspectRatio)             flag |= CV_CALIB_FIX_ASPECT_RATIO;
    }

    // Sets up the next image for pattern detection
    Mat imageSetup(int imageIndex)
    {
        Mat img;
        if( imageIndex < (int)imageList.size() )
	img = imread(imageList[imageIndex], CV_LOAD_IMAGE_COLOR);

	/*
	if (img.data){

	  Point2f img_center(img.cols/2.0F, img.rows/2.0F);
	  Mat rot_mat = getRotationMatrix2D(img_center, 180, 1.0);
	  //warpAffine(img, img, rot_mat, img.size());

	  rotate(img, img, ROTATE_180);
	}
	*/

        return img;
    }

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

    
 
    // Sets up intrinsicInput struct from an intrinsics file
    bool readIntrinsicInput( const string& filename )
    {
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened() ) {
            if ( filename == "0" )       // Intentional lack of input
                return false;
            else {                       // Unintentional invalid input
                cout << "Invalid intrinsic input: " << filename << endl;
                return false;
            }
        }
        fs["Camera_Matrix"] >> intrinsicInput.cameraMatrix;
        fs["Distortion_Coefficients"] >> intrinsicInput.distCoeffs;
        return true;
    }

    // Saves the intrinsic parameters of the inCal struct to intrinsicOutput
    void saveIntrinsics(intrinsicCalibration &inCal)
    {
        if (intrinsicOutput == "0") return;
        FileStorage fs( intrinsicOutput, FileStorage::WRITE );

        time_t tm;
        time( &tm );
        struct tm *t2 = localtime( &tm );
        char buf[1024];
        strftime( buf, sizeof(buf)-1, "%c", t2 );
        fs << "Calibration_Time" << buf;
	
        fs << "Image_Width" << imageSize.width;
        fs << "Image_Height" << imageSize.height;

        fs << "Calibration_Pattern" << patternInput;
        if (calibrationPattern == CHESSBOARD)
        {
            fs << "Board_Width" << boardSize.width;
            fs << "Board_Height" << boardSize.height;
            fs << "Square_Size" << squareSize;
        }

        if( flag & CV_CALIB_FIX_ASPECT_RATIO )
            fs << "AspectRatio" << aspectRatio;

        if( flag )
            sprintf( buf, "%s%s%s%s%s%s%s%s%s",
                flag & CV_CALIB_FIX_K1 ? "+FIX_K1 " : "",
                flag & CV_CALIB_FIX_K2 ? "+FIX_K2 " : "",
                flag & CV_CALIB_FIX_K3 ? "+FIX_K3 " : "",
                flag & CV_CALIB_FIX_K4 ? "+FIX_K4 " : "",
                flag & CV_CALIB_FIX_K5 ? "+FIX_K5 " : "",
                flag & CV_CALIB_USE_INTRINSIC_GUESS ? "+USE_INTRINSIC_GUESS " : "",
                flag & CV_CALIB_FIX_ASPECT_RATIO ? "+FIX_ASPECT_RATIO " : "",
                flag & CV_CALIB_FIX_PRINCIPAL_POINT ? "+FIX_PRINCIPAL_POINT " : "",
                flag & CV_CALIB_ZERO_TANGENT_DIST ? "+ZERO_TANGENT_DIST " : "" );
        fs << "Calibration_Flags" << buf;

        fs << "Camera_Matrix" << inCal.cameraMatrix;
        fs << "Distortion_Coefficients" << inCal.distCoeffs;

        fs << "Avg_Reprojection_Error" << inCal.totalAvgErr;
        if( !inCal.reprojErrs.empty() )
            fs << "Per_View_Reprojection_Errors" << Mat(inCal.reprojErrs);
    }

    // Saves the stereo parameters of the sterCal struct to extrinsicOutput
    void saveExtrinsics(stereoCalibration &sterCal)
    {
        if (extrinsicOutput == "0") return;
        FileStorage fs( extrinsicOutput, FileStorage::WRITE );

        time_t tm;
        time( &tm );
        struct tm *t2 = localtime( &tm );
        char buf[1024];
        strftime( buf, sizeof(buf)-1, "%c", t2 );
        fs << "Calibration_Time" << buf;

        fs << "Calibration_Pattern" << patternInput;

        fs << "Stereo_Parameters";
        fs << "{" << "Rotation_Matrix"     << sterCal.R
                  << "Translation_Vector"  << sterCal.T
                  << "Essential_Matrix"    << sterCal.E
                  << "Fundamental_Matrix"  << sterCal.F
           << "}";

        fs << "Rectification_Parameters";
        fs << "{" << "Rectification_Transformation_1"       << sterCal.R1
                  << "Rectification_Transformation_2"       << sterCal.R2
                  << "Projection_Matrix_1"                  << sterCal.P1
                  << "Projection_Matrix_2"                  << sterCal.P2
                  << "Disparity-to-depth_Mapping_Matrix"    << sterCal.Q
           << "}";
    }

public:

  bool goodInput;         //Tracks input validity
  
//--------------------------Calibration configuration------------------------------//
    // Program modes:
    //    INTRINSIC  — calculates intrinsics parameters and  undistorts images
    //    STEREO     — calculates extrinsic stereo paramaters and rectifies images 
  Mode mode;
  Pattern calibrationPattern;   // Three supported calibration patterns: CHESSBOARD, ARUCO_SINGLE, ARUCO_BOX
  
//-----------------------------AruCo configuration---------------------------------//
  
     
  vector<int>  markersX;       // Number of AruCo Markers in first row
  vector<int>  markersY;       // Number of AruCo Markers in first column 
  vector<float> markerLength;  // The length of the markers in pixels
  int dictionary;              // The number of the AruCo dictionary used to draw the markers
  vector< int > type;          // The id of the first marker of the board
  int numberOfBoards;          // Number of boards in the scene. Default:1


//--------------------------Chessboard configuration--------------------------//
  Size boardSize;     // Size of chessboard (number of inner corners per chessboard row and column)
  float squareSize;   // The size of a square in some user defined metric system (pixel, millimeter, etc.)
  
//-----------------------------Input settings---------------------------------//
  vector<string> imageList;   // Image list to run calibration
  string imageListFilename;   // Input filename for image list
 
  //Intrinsic input can be used as an initial estimate for intrinsic calibration,
  //as fixed intrinsics for stereo calibration.
  //Leave filename at "0" to calculate new intrinsics
  intrinsicCalibration intrinsicInput; // Struct to store inputted intrinsics
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

// Tommaso Monaco, Summer 2018 
//----------------------Stereo Calibration settings---------------------------//

  int alpha; // Free scaling factor during rectification
  int rf;    // Resizing factor after rectification:
             //  decides how large the resolution of the
             //  image after rectification with respect to the original
             //  image size.
  
  // LEAVE THIS VALUE AT "0" TO NOT CROP RECTIFIED IMAGES
  int crop;  // Crops the final rectified images
  

//--------------------------------UI settings---------------------------------//
    bool showUndistorted;   // Show undistorted images after intrinsic calibration
    bool showRectified;     // Show rectified images after stereo calibration
    bool wait;              // Wait until a key is pressed to show the next detected image

//-----------------------------Program variables------------------------------//
    int nImages;        // Number of images in the image list
    Size imageSize;     // Size of each image
    int nBoards;       // Number of marker maps read from config list

private:
    // Input variables only needed to set up settings
    string modeInput;
    string patternInput;
};

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
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


Ptr<ChessBoard> ChessBoard::create(int markersX, int markersY, float markerLength,
                                   float markerSeparation,
                                   const Ptr<aruco::Dictionary> &dictionary, int firstMarker) {

    Ptr<ChessBoard> res = makePtr<ChessBoard>();

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
bool pathCheck(const string& path)
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
void printMat(Mat m, const char *name)
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
void printPoints(const intrinsicCalibration inCal)
{
    for (auto v:inCal.objectPoints)
    {
        cout << "object " << v.size() << endl << "[";
        for (auto p:v)
             cout << " " << p << " ";
        cout << endl << endl;
    }
    
    for (auto v:inCal.imagePoints)
    {
        cout << "image " << v.size() << endl << "[";
        for (auto p:v)
             cout << " " << p << " ";
        cout << endl << endl;
    }
    
}


//-------------------------Calibration functions------------------------------//
// Calculates the reprojection error with a set of intrinsics
double computeReprojectionErrors(intrinsicCalibration &inCal)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    inCal.reprojErrs.resize(inCal.objectPoints.size());
    for( i = 0; i < (int)inCal.objectPoints.size(); i++ )
    {
        projectPoints(Mat(inCal.objectPoints[i]), inCal.rvecs[i], inCal.tvecs[i],
                      inCal.cameraMatrix, inCal.distCoeffs, imagePoints2);
        err = norm(Mat(inCal.imagePoints[i]), Mat(imagePoints2), CV_L2);
        int n = (int)inCal.objectPoints[i].size();
        inCal.reprojErrs[i] = (float)sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }
    return sqrt(totalErr/totalPoints);
}

// Calculates the 3D object points of a chessboard
void calcChessboardCorners(Settings s, vector<Point3f>& objectPointsBuf)
{
    for( int i = 0; i < s.boardSize.height; i++ )
        for( int j = 0; j < s.boardSize.width; j++ )
            objectPointsBuf.push_back(Point3f(float(j*s.squareSize),
                                      float(i*s.squareSize), 0));
}


// Detects the pattern on a chessboard image
void chessboardDetect(Settings s, Mat &img, intrinsicCalibration &inCal)
{
    //create grayscale copy for cornerSubPix function
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    //buffer to store points for each image
    vector<Point2f> imagePointsBuf;
    vector<Point3f> objectPointsBuf;

    bool found = findChessboardCorners( img, s.boardSize, imagePointsBuf,
        CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS | CV_CALIB_CB_FAST_CHECK |
        CV_CALIB_CB_NORMALIZE_IMAGE);

    
    if (found)
    {
        cornerSubPix(imgGray, imagePointsBuf, Size(11,11), Size(-1,-1),
                     TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

        //add these image points to the overall calibration vector
        inCal.imagePoints.push_back(imagePointsBuf);

        //find the corresponding objectPoints
        calcChessboardCorners(s, objectPointsBuf);
        inCal.objectPoints.push_back(objectPointsBuf);
        drawChessboardCorners(img, s.boardSize, Mat(imagePointsBuf), found);
    }
}




// Tommaso Monaco, Summer 2018
//-------------------------Aruco Detection and Processing Functions----------------------//

// Stereo calibration requires both images to have the same # of image and object points, but
//  ArUco detections can include an arbitrary subset of all markers.
//  This function limits the points lists to only those points shared between each image
void getSharedPoints(intrinsicCalibration &inCal, intrinsicCalibration &inCal2) {
  
  vector<Point3f> *oPoints, *oPoints2; // pointers to make code more legible
  vector<Point2f> *iPoints, *iPoints2;
  int shared;                          //index of a shared object point
  bool paddingPoints = false;          //padding points in case no object points were detected
    
  //for each objectPoints vector in overall objectPoints vector of vectors
  for (int i=0; i< (int)inCal.objectPoints.size(); i++) {
    
    vector<Point3f> sharedObjectPoints;
    vector<Point2f> sharedImagePoints, sharedImagePoints2; //shared image points for each inCal

    oPoints = &inCal.objectPoints.at(i); // obj points from view1 
    oPoints2 = &inCal2.objectPoints.at(i);
    iPoints  = &inCal.imagePoints.at(i); // obj points from view2
    iPoints2 = &inCal2.imagePoints.at(i);

    // If the number of obj points from view1 is larger than
    //  the number of object point from view2
    if ((int)oPoints->size() >= (int)oPoints2->size()){ 
      for (int j=0; j<(int)oPoints->size(); j++) {
	if (oPoints->at(0) == Point3f(-1,-1,0)) {
	  paddingPoints = true; 
	  break;
	}	
	for (shared=0; shared<(int)oPoints2->size(); shared++)
	  if (oPoints->at(j) == oPoints2->at(shared)) break;
	if (shared != (int)oPoints2->size()){  //obj point is shared
	  
	  sharedObjectPoints.push_back(oPoints->at(j)); 
	  sharedImagePoints.push_back(iPoints->at(j));
	  sharedImagePoints2.push_back(iPoints2->at(shared));
	}
	paddingPoints = false;
      }
    }
    
    // If the number of obj points from view2 is larger than
    //  the number of object point from view1
    else {
      for (int j=0; j<(int)oPoints2->size(); j++) {
	if (oPoints2->at(0) == Point3f(-1,-1,0)) {		
	  paddingPoints = true;
	  break;
	}	      
	for (shared=0; shared<(int)oPoints->size(); shared++)
	  if (oPoints2->at(j) == oPoints->at(shared)) break;
	if (shared != (int)oPoints->size()) {   //obj point is shared

	  sharedObjectPoints.push_back(oPoints2->at(j));
	  sharedImagePoints2.push_back(iPoints2->at(j));
	  sharedImagePoints.push_back(iPoints->at(shared));
	}	      
	paddingPoints = false;
      }
    }

    // shared obj points are included in the final calibration structures.
    if ((int) sharedObjectPoints.size() > 10){
      *oPoints = sharedObjectPoints;
      *oPoints2 = sharedObjectPoints;
      *iPoints = sharedImagePoints;
      *iPoints2 = sharedImagePoints2;	
    }

    // if the number of shared obj points is too low, do not include the obj points
    //  in the final calibration structure.
    else if ((int) sharedObjectPoints.size() < 10 || paddingPoints) { 
      
      inCal.objectPoints.erase(inCal.objectPoints.begin()+i);
      inCal2.objectPoints.erase(inCal2.objectPoints.begin()+i);
      inCal.imagePoints.erase(inCal.imagePoints.begin()+i);
      inCal2.imagePoints.erase(inCal2.imagePoints.begin()+i);
      
      // if no obj points are left to be considered, then break from loop already
      if (inCal.objectPoints.size() <= 0) {
	inCal.objectPoints[0].clear();
	inCal2.objectPoints[0].clear();
	break;
      }
	  
      // decrement i; one element
      //  was removed from beginning of
      //  overall objectPoints vector of vectors
      i--; 
    }
  }   
}





void getObjectAndImagePoints( vector< vector< Point2f > >  detectedCorners, vector< int > detectedIds, vector< Point3f > &objPoints, vector< Point2f > &imgPoints, Ptr<ChessBoard> &currentBoard) {

    std::map< string , int> countMap; //count of occurances of each obj point.
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
		  //check if the obj point is a duplicate
		  auto result = countMap.insert(std::pair< string , int>(temp.str() , 1));
		  if (result.second == false)
		    result.first-> second++;
		  if (result.second !=1) //if duplicate, do not consider the obj point.
		    continue;
		  objPoints.push_back(currentBoard->obj_points_vector[j][p]);
		  imgPoints.push_back(detectedCorners[i][p]);
                    
                }
            }
        }
    }
}


void processPoints(Settings s, vector< vector< Point2f > > corners, vector<int> ids,
		   vector<int> counter, vector< vector < Point2f >> &processedImagePoints,
		   vector< vector<Point3f>> &processedObjectPoints, Ptr<ChessBoard> &currentBoard) {

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

    getObjectAndImagePoints(thisFrameCorners, thisFrameIds, currentObjPoints,
                            currentImgPoints, currentBoard);
    
				       
    if(currentImgPoints.size() > 0 && currentObjPoints.size() > 0) {
      processedImagePoints.push_back(currentImgPoints);
      processedObjectPoints.push_back(currentObjPoints);
    }
    
    else if (currentImgPoints.size() <= 0 && currentObjPoints.size() <= 0
	     && s.mode == Settings::STEREO) {
      
      for (int i=0; i < 4; i++){
	currentImgPoints.push_back(Point2f(-1,-1));
	currentObjPoints.push_back(Point3f(-1,-1,0));
      }
      processedImagePoints.push_back(currentImgPoints);
      processedObjectPoints.push_back(currentObjPoints);
    }
  }
}

void setupAruco( Settings s, arucoCalibration &arCal, arucoCalibration &arCal2,
		 Ptr<ChessBoard> &currentBoard){

  // concatenates all detected corners in an ArrayOfArrays (allCornersConcatenated1).
  // concatenates the correspoding ids in an ArrayOfArrays (allIdsConcatenated1).
  // markerCounterPerFrame1 counts the number of detected corners
  //  in each frame.
  arCal.markerCounterPerFrame.reserve(arCal.allCorners.size());
  for(unsigned int i = 0; i < arCal.allCorners.size(); i++) {
    arCal.markerCounterPerFrame.push_back((int)arCal.allCorners[i].size());
    for(unsigned int j = 0; j < arCal.allCorners[i].size(); j++) {
      arCal.allCornersConcatenated.push_back(arCal.allCorners[i][j]);
      arCal.allIdsConcatenated.push_back(arCal.allIds[i][j]);
    }
  }
  
  vector< vector < Point2f >>  processedImagePoints1;
  vector< vector<Point3f>> processedObjectPoints1 ;
  
  // prepares data for aruco calibration  
  processPoints(s, arCal.allCornersConcatenated,
		arCal.allIdsConcatenated, arCal.markerCounterPerFrame,
		processedImagePoints1, processedObjectPoints1, currentBoard);
  
  arCal.objPoints = processedObjectPoints1;
  arCal.imgPoints = processedImagePoints1;
  
  // If stereo mode, repeat the process for the second viewpoint
  if(s.mode == Settings::STEREO){
    
    arCal2.markerCounterPerFrame.reserve(arCal2.allCorners.size());
    for(unsigned int i = 0; i < arCal2.allCorners.size(); i++) {
      arCal2.markerCounterPerFrame.push_back((int)arCal2.allCorners[i].size());
      for(unsigned int j = 0; j < arCal2.allCorners[i].size(); j++) {
	arCal2.allCornersConcatenated.push_back(arCal2.allCorners[i][j]);
	arCal2.allIdsConcatenated.push_back(arCal2.allIds[i][j]);
      }
    }
    
    vector< vector < Point2f >>  processedImagePoints2;
    vector< vector<Point3f>> processedObjectPoints2;
    
    processPoints(s, arCal2.allCornersConcatenated,
		  arCal2.allIdsConcatenated, arCal2.markerCounterPerFrame,
		  processedImagePoints2, processedObjectPoints2, currentBoard);
    
    arCal2.objPoints = processedObjectPoints2;
    arCal2.imgPoints = processedImagePoints2;
    
  }
  
  
}


void drawDetected(Mat &image, vector< vector< Point2f > > corners,  vector< int > ids) {
  
  // calculate colors
  Scalar textColor, cornerColor, borderColor;
  borderColor = Scalar(51, 153, 255);
  cornerColor = Scalar(255, 0, 127);
  textColor = Scalar(0, 0, 0);
  
  int nMarkers = (int)corners.size();
  for(int i = 0; i < nMarkers; i++) {
    vector< Point2f > currentMarker = corners[i];
    CV_Assert(currentMarker.size() == 4);


    // Draws a square on the current marker
    vector<Point> pt;
    for (int ao=0; ao< 4; ao++) {
      pt.push_back(currentMarker[ao]);
    } 
    fillConvexPoly(image, pt, borderColor);

    
    // Draw the marker ID
    if(ids.size() != 0) {
      Point2f cent(0, 0);
      for(int p = 0; p < 4; p++)
	cent += currentMarker[p];
      cent = cent / 4.;
      putText(image,std::to_string(ids[i]), cent, FONT_HERSHEY_DUPLEX, 0.3, textColor, 1.9);
    }
    
  }

  // Highlight the top right-hand corner of the current marker
  //  (within a second loop to avoid superposition of shapres)
  for(int i = 0; i < nMarkers; i++) {
    vector< Point2f > currentMarker = corners[i];
    rectangle(image, currentMarker[0] - Point2f(3, 3),
		  currentMarker[0] + Point2f(3, 3), cornerColor, 2, LINE_AA);
  }
}


Mat arucoDetect(Settings s, Mat &img, arucoCalibration &arCal, Ptr<ChessBoard> currentBoard, bool &letsBreak){

  Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

  //detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; //OpenCV 3.3 and above.
  detectorParams-> doCornerRefinement = true; // do corner refinement in markers // OpenCV 3.2 and below
  detectorParams-> cornerRefinementWinSize = 4;
  detectorParams->  minMarkerPerimeterRate = 0.01;
  detectorParams->  maxMarkerPerimeterRate = 4 ; 
  detectorParams-> cornerRefinementMinAccuracy = 0.05;
  
  
  Mat imgCopy;
  
  // detect markers
  aruco::detectMarkers(img, currentBoard->localDictionary,
                       currentBoard->corners, currentBoard->ids,
                       detectorParams, currentBoard->rejected);
  
  // draw results
  img.copyTo(imgCopy);
  if(currentBoard->ids.size() > 0) drawDetected(imgCopy, currentBoard->corners,
						currentBoard->ids);

  
  putText(imgCopy, "Press 'c' to move to the next frame. 'ESC' to finish and calibrate",
          Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
  
    
  imshow( "Detected" , imgCopy);
  char key = (char)waitKey(100000);
  if (key == 27) 
    letsBreak = true;
  
  else if (key == 'c' || currentBoard->ids.size() > 0) {  
    arCal.allCorners.push_back(currentBoard->corners); 
    arCal.allIds.push_back(currentBoard->ids);
    s.imageSize = img.size();
  }
  
  else if(key == 'c' && currentBoard->ids.size() == 0 && s.mode == Settings::STEREO) {
    vector < Point2f > temp;  //if no detected, use temporary padding points, e.g. (-1,-1,0
    for (int i=0 ; i < 4; i++)   
      temp.push_back(Point2f(-1,-1));
    
    currentBoard->corners.push_back(temp);
    currentBoard->ids.push_back(-1);

    arCal.allCorners.push_back(currentBoard->corners);
    arCal.allIds.push_back(currentBoard->ids);
  }
  
  return imgCopy;
  
}




//--------------------Running and saving functions----------------------------//
// Correct an images radial distortion using a set of intrinsic parameters
static void undistortImages(Settings s, intrinsicCalibration &inCal)
{
    Mat img, Uimg;
    char imgSave[1000];

    bool save = false;
    
    if(s.undistortedPath != "0")
    {
        if( pathCheck(s.undistortedPath) )
            save = true;
        else
            printf("\nUndistorted images could not be saved. Invalid path: %s\n", s.undistortedPath.c_str());
    }

    namedWindow("Undistorted", CV_WINDOW_AUTOSIZE);
    for( int i = 0; i < s.nImages; i++ )
    {
        img = s.imageSetup(i);
        undistort(img, Uimg, inCal.cameraMatrix, inCal.distCoeffs);

        // If a valid path for undistorted images has been provided, save them to this path
        if(save)
        {
            sprintf(imgSave, "%sundistorted_%d.jpg", s.undistortedPath.c_str(), i);
            imwrite(imgSave, Uimg);
        }

        if(s.showUndistorted)
        {
            imshow("Undistorted", Uimg);
            char c = (char)waitKey();
            if( (c & 255) == 27 || c == 'q' || c == 'Q' )   //escape key or 'q'
                break;
        }
    }
    destroyWindow("Undistorted");
}


vector<int> cropImage( vector<Mat> rectifiedPair,  vector<vector<int>> roi,   int pair){

  vector<Mat> croppedImages;
  vector<int> dfs;
  char imgSave[100];
  bool save =  false;
  
		        
  int y = min(max(roi[0][1]- 100, 0), max(roi[1][1]- 100, 0)); // new y-coordinate for new cropped image pair

  //  Compute width and height of image pair using region of interest.
  //   The final cropped pair needs to have the same width and height.
  int y1 = max(min( (roi[0][1] + roi[0][3])+ 100 , rectifiedPair[0].size().height - roi[0][1]),
	       min( (roi[1][1] + roi[1][3])+ 100,  rectifiedPair[1].size().height - roi[1][1]));
  int x = min(max(roi[0][0] - 100, 0) , max(roi[1][0] - 100, 0));
  int x1 = max(min( (roi[0][0] + roi[0][2])+ 100, rectifiedPair[0].size().width - roi[0][0]),
	       min( (roi[1][0] + roi[1][2])+ 100,  rectifiedPair[1].size().width - roi[1][0]));

  // Width and height
  int w = x1 - x;
  int h = y1 - y;


  for ( int view =0; view < 2; view++) {
    
    
    int x = max(roi[view][0] - 100, 0); // new x-cordinate for each of the new cropped images

    dfs.push_back(x); 
    
    Rect mask(x,y,w,h);
    Mat croppedImage = rectifiedPair[view](mask);
    
    croppedImages.push_back(croppedImage);

    if (save)
      {
	sprintf(imgSave, "cropped-%d-%d.jpg", pair, view);
	imwrite(imgSave, croppedImages[view]);  
      }
  }
 
  dfs.push_back(y); // return vector (dfs) contains:
                    //  - dx after cropping of first image in image pair
                    //  - dx after cropping of second image in image pair
                    //  - dy after cropping for both images in image pair
  
  return dfs;
  
}

// Threhold image after rectification to obtain region of interest - i.e. ignore black background and noise.
vector<int> thresholdImage(stereoCalibration sterCal, Mat rectifiedImage, int view , int pair){

  // structures for saving images
  char imgSave[100]; 
  bool save = false;

  // structures for thresholding
  Mat src_gray;
  int thresh = 1;
  int max_thresh = 255;
  Mat threshold_output;
  
  // Clone and blur rectified image before thresholding
  cvtColor(rectifiedImage, src_gray, COLOR_BGR2GRAY);
  
  /// Threshold grayscale image
  threshold( src_gray, threshold_output, thresh, max_thresh, THRESH_BINARY );

  
  Mat out_color;
  Mat labeledImage;
  Mat stats;
  Mat centroids;
  int nLabels;
  int connectivity = 4;

  // an estimate for the bounding box of the desired frame
  int valPixelsArea = sterCal.validRoi[view].height *  sterCal.validRoi[view].width;

  // vector that will contain indeces to the largest
  //  and second largest connected component in the image
  vector<int> ccVector;
  ccVector.push_back(1); // largest
  ccVector.push_back(2); // second largest

    
  /// Find connected componets
  nLabels =  connectedComponentsWithStats(threshold_output, labeledImage,
					  stats, centroids, connectivity, CV_32S);

  //vector of colors for each connected component
  vector<cv::Vec3b> colors(nLabels+1);
  
  for (int j = 0; j < 2 ; j++)
    for (int l = 1; l < nLabels; l++)
      {

	int w = stats.at<int>(Point(2, l));
	int h = stats.at<int>(Point(3, l));
	int a = stats.at<int>(Point(4, l));

	
	colors[l] = cv::Vec3b(0,0,0); // background pixels remain black.

	//if larger than the region of valid pixels, ignore the  connected compoent
	if (w > sterCal.validRoi[view].width*2 || h >  sterCal.validRoi[view].width*2)
	  continue;
	
	if (a > valPixelsArea && a >= stats.at<int>(Point(4, ccVector[j])))
	  {
	    if (j==1 && ccVector[j-1]==l)
	      continue;
	    ccVector[j] = l;
	  }
      }


  int lcc = ccVector[0];
  int slcc = ccVector[1];

  
  //think if there is a better way than this ...
  if (stats.at<int>(Point(3, lcc)) > stats.at<int>(Point(3, slcc)) &&
      stats.at<int>(Point(2, lcc)) < sterCal.validRoi[view].width) {
    ccVector[0]  = slcc;
    ccVector[1] = lcc;
  }
  

  // the connected component of interest will be white, (255, 255, 255)
  colors[ccVector[0]] = cv::Vec3b(255, 255, 255);
  out_color  = cv::Mat::zeros(rectifiedImage.size(), CV_8UC3);
  for( int y = 0; y < out_color.rows; y++ )
    for( int x = 0; x < out_color.cols; x++ )
      {
	int label = labeledImage.at<int>(y, x);
	out_color.at<cv::Vec3b>(y, x) = colors[label];
      }

  int lx = stats.at<int>(Point(0, ccVector[0]));
  int ly = stats.at<int>(Point(1, ccVector[0]));
  int lw = stats.at<int>(Point(2, ccVector[0]));
  int lh = stats.at<int>(Point(3, ccVector[0]));
  int la = stats.at<int>(Point(4, ccVector[0]));

  vector<int> roi = {lx, ly, lw, lh, la};
  
  if (save)
    {
      sprintf(imgSave, "masked-%d-%d.jpg", pair, view);
      imwrite(imgSave, out_color);  
    }
  
  return roi;
}

   

// Rectifies an image pair using a set of extrinsic stereo parameters
void rectifyImages(Settings s, intrinsicCalibration &inCal,
                   intrinsicCalibration &inCal2, stereoCalibration &sterCal)
{
    Mat rmap[2][2];

    //Precompute maps for remap()
    initUndistortRectifyMap(inCal.cameraMatrix, inCal2.distCoeffs, sterCal.R1,
			    sterCal.P1, s.imageSize * s.rf, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(inCal2.cameraMatrix, inCal2.distCoeffs, sterCal.R2,
                        sterCal.P2, s.imageSize* s.rf, CV_16SC2, rmap[1][0], rmap[1][1]);
    
    
    Mat canvas, rimg, cimg;    
    double sf = 600. /  MAX(s.imageSize.width, s.imageSize.height);
    int w = cvRound(s.imageSize.width * sf);
    int h = cvRound(s.imageSize.height * sf);
    canvas.create(h, w*2, CV_8UC3);
    
    // buffer for image filename
    char imgSave[1000];
    const char *view;
    
    vector<int> dfs;   // if 'crop' is activated,
                       // contains how much image was cropped
                       // horizontally and vertically

    bool save = false;
    if(s.rectifiedPath != "0")
    {
        if( pathCheck(s.rectifiedPath) )
            save = true;
        else
            printf("\nRectified images could not be saved. Invalid path: %s\n", s.rectifiedPath.c_str());
    }
   
    namedWindow("Rectified", CV_WINDOW_AUTOSIZE);
    for( int i = 0; i < s.nImages/2 ; i++ ) {

      vector<Mat> vectorRimgs;
      vector<vector<int>> vectorROIs;
      
      for( int k = 0; k < 2; k++ ) {
	  
	  Mat img = s.imageSetup(i*2+k), rimg, cimg;	  	  
	  remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
	    
	  // If a valid path for rectified images has been provided,
	  //  save them to this path
            if (save)
            {
	      if (s.crop == 1){
		vector<int> roi  = thresholdImage(sterCal, rimg, k, i);
		vectorROIs.push_back(roi);
		vectorRimgs.push_back(rimg);
	      }
	      
	      view = "left";
	      if (k == 1) view = "right";
	      sprintf(imgSave, "%s%s_rectified_%d.jpg", s.rectifiedPath.c_str(), view, i);
	      imwrite(imgSave, rimg);
            }

          
            Mat canvasPart = canvas(Rect(w*k, 0, w, h));
	    resize(rimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
	    
            Rect vroi(cvRound(sterCal.validRoi[k].x * sf * 1/s.rf),
		      cvRound(sterCal.validRoi[k].y * sf * 1/s.rf),
                      cvRound(sterCal.validRoi[k].width * sf * 1/s.rf),
		      cvRound(sterCal.validRoi[k].height * sf * 1/s.rf)); 
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
	    
        }

      for( int j = 0; j < canvas.rows; j += 16 )
	  line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

      if (s.showRectified) {
	imshow("Rectified", canvas);
	char c = (char)waitKey(10000);
	if( c == 27 || c == 'q' || c == 'Q' )
	  break;
      }
      
      if (s.crop == 1) // if 'crop', then crop img pair
	dfs =cropImage(vectorRimgs, vectorROIs, i);
      
    }

    if (s.crop == 1){// if 'crop', then update projection matrices 
      
      sterCal.P1.at<double>(0,2) = sterCal.P1.at<double>(0,2) - dfs[0];
      sterCal.P2.at<double>(0,2) = sterCal.P2.at<double>(0,2) - dfs[1];
      sterCal.P1.at<double>(1,2) = sterCal.P1.at<double>(1,2) - dfs[2];
      sterCal.P2.at<double>(1,2) = sterCal.P2.at<double>(1,2) - dfs[2];
    }
    
    destroyWindow("Rectified");
}

// Run intrinsic calibration, using the image and object points to calculate the
// camera matrix and distortion coefficients
bool runIntrinsicCalibration(Settings s, intrinsicCalibration &inCal)
{
    if (s.useIntrinsicInput)     //precalculated intrinsic have been inputted. Use these
    {
        inCal.cameraMatrix = s.intrinsicInput.cameraMatrix;
        inCal.distCoeffs = s.intrinsicInput.distCoeffs;
        calibrateCamera(inCal.objectPoints, inCal.imagePoints, s.imageSize,
                        inCal.cameraMatrix, inCal.distCoeffs,
                        inCal.rvecs, inCal.tvecs, s.flag | CV_CALIB_USE_INTRINSIC_GUESS);

    } else {                //else, create empty matrices to be calculated
        inCal.cameraMatrix = Mat::eye(3, 3, CV_64F);
        inCal.distCoeffs = Mat::zeros(8, 1, CV_64F);
	  
        cout << inCal.objectPoints.size()<< endl;
	cout << inCal.imagePoints.size()<< endl;
	
        calibrateCamera(inCal.objectPoints, inCal.imagePoints, s.imageSize,
                        inCal.cameraMatrix, inCal.distCoeffs,
                        inCal.rvecs, inCal.tvecs, s.flag);

	cout << inCal.cameraMatrix << endl;
	cout << inCal.distCoeffs << endl;
    }

     
    bool ok = checkRange(inCal.cameraMatrix) && checkRange(inCal.distCoeffs);   
    inCal.totalAvgErr = computeReprojectionErrors(inCal);
    
    return ok;
}

// Run stereo calibration, using the points and intrinsics of two viewpoints to determine
// the rotation and translation between them
stereoCalibration runStereoCalibration(Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2)
{
    stereoCalibration sterCal;
    if (s.useIntrinsicInput) {    //precalculated intrinsic have been inputted. Use these
      
      inCal.cameraMatrix = s.intrinsicInput.cameraMatrix;
      inCal2.cameraMatrix = s.intrinsicInput.cameraMatrix;
      inCal.distCoeffs = s.intrinsicInput.distCoeffs;
      inCal2.distCoeffs = s.intrinsicInput.distCoeffs;
    }

    
    // Stereo calibration requires both images
  //  to have the same # of image and object points;
  //  getSharedPoints limits the points lists
  //  to only those points shared between each image
  if (s.calibrationPattern != Settings::CHESSBOARD)      //ArUco pattern
    getSharedPoints(inCal, inCal2);
  
  double err = stereoCalibrate(inCal.objectPoints, inCal.imagePoints, inCal2.imagePoints,
			       inCal.cameraMatrix, inCal.distCoeffs,
			       inCal2.cameraMatrix, inCal2.distCoeffs,
			       s.imageSize, sterCal.R, sterCal.T, sterCal.E, sterCal.F,
			       CV_CALIB_FIX_INTRINSIC,
			       TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-10));

    printf("\nStereo reprojection error = %.4f\n", err);

    
    stereoRectify(inCal.cameraMatrix, inCal.distCoeffs,
		  inCal2.cameraMatrix, inCal2.distCoeffs,
		  s.imageSize, sterCal.R, sterCal.T, sterCal.R1, sterCal.R2,
		  sterCal.P1, sterCal.P2, sterCal.Q,
		  CALIB_ZERO_DISPARITY, s.alpha,  s.imageSize * s.rf,
		  &sterCal.validRoi[0], &sterCal.validRoi[1]);
    
    rectifyImages(s, inCal, inCal2, sterCal);
    return sterCal;
}

// Runs the appropriate calibration based on the mode and saves the results
void runCalibrationAndSave(Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2)
{
    bool ok;
    if (s.mode == Settings::STEREO) {         // stereo calibration
        if (!s.useIntrinsicInput)
        {
	  
	  // Stereo calibration requires both images to have the same # of image and object points;
	  // getSharedPoints limits the points lists to only those points shared between each image
	  if (s.calibrationPattern != Settings::CHESSBOARD) {     //ArUco pattern
	    getSharedPoints(inCal, inCal2);
	  }

	  ok = runIntrinsicCalibration(s, inCal);	  
	  printf("%s for left. Avg reprojection error = %.4f\n",
                ok ? "\nIntrinsic calibration succeeded" : "\nIntrinsic calibration failed",
                inCal.totalAvgErr);
	  
	  
	  ok = runIntrinsicCalibration(s, inCal2);
	  printf("%s for right. Avg reprojection error = %.4f\n",
		 ok ? "\nIntrinsic calibration succeeded" : "\nIntrinsic calibration failed",
		 inCal2.totalAvgErr);
        } else
            ok = true;
	
        stereoCalibration sterCal = runStereoCalibration(s, inCal, inCal2);
        s.saveExtrinsics(sterCal);

	
    } else {                        // intrinsic calibration
      ok = runIntrinsicCalibration(s, inCal);
      printf("%s. Avg reprojection error = %.4f\n",
	     ok ? "\nIntrinsic calibration succeeded" : "\nIntrinsic calibration failed",
	     inCal.totalAvgErr);
      
      if( ok ) {
	undistortImages(s, inCal);
	s.saveIntrinsics(inCal);
      }
    }
}


// Main function. Detects patterns on images, runs calibration and saves results
int calibrateWithSettings( const string inputSettingsFile )
{

  
    Settings s;
    FileStorage fs(inputSettingsFile, FileStorage::READ);   // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the settings file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    
    // helper variable for breaking the loops
    bool letsBreak = false;
    
    // struct to store calibration parameters
    intrinsicCalibration inCal, inCal2;
    intrinsicCalibration *currentInCal = &inCal;
    
    // size for stereo calibration
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


    
/*-----------Calibration using AruCo patterns--------------*/ 
    if (s.calibrationPattern != Settings::CHESSBOARD) {
    
    // These structures are stored into lists (based on the # of boards in the scene)
    vector< Ptr<ChessBoard> > boardsList;
    vector< vector < arucoCalibration > > arCalList;
    
    // The AruCo dictionary used. 
    Ptr<aruco::Dictionary> dictionary =
      aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(s.dictionary));     
    
    
    
    // initialize the AruCo structures
    for(int n = 0; n< s.numberOfBoards; n++){

	Ptr<ChessBoard> board = ChessBoard::create(s.markersX[n], s.markersY[n],
						   s.markerLength[n], s.markerLength[n],
						   dictionary, s.type[n]);
	boardsList.push_back(board);
	
	// struct for storing aruCo parameters.
	//  One for each stereo viewpoint.
	arucoCalibration arCal, arCal2;
	
	// require correctly sized vectors
	arCal.imgPoints.resize(size);
	arCal.objPoints.resize(size);
	arCal2.imgPoints.resize(size);
	arCal2.objPoints.resize(size);
	
	vector < arucoCalibration > tempList;
	// make a list of lists of the newly created intrinsicCalibration structs;
	//  the size of the list corresponds to the number of boards used. 
	tempList.push_back(arCal);
	tempList.push_back(arCal2);
	arCalList.push_back(tempList);
      }
      
      
      // variable used to facilitate the alternation
      //  between intrinsic calibration structs for stereo mode.
      int currentView = 0;
      arucoCalibration *currentArCal;
    
      
      for(int i = 0;;i++) {
	// Switches between intrinsic calibration structs for stereo mode
	if (i%2 == 0) 
	  currentView = 0;
	else if (s.mode == Settings::STEREO)
	  currentView = 1;
	
	// Set up the image
	Mat img = s.imageSetup(i);
	
	
	if(!img.data)
	  {
	    destroyWindow("Detected");
	    
	    for(int n = 0; n< s.numberOfBoards; n++){
		
	      setupAruco(s, arCalList[n][0], arCalList[n][1], boardsList[n]);
		
	      // inCal is the final structure used for the calibration.
	      //  Thus, move all the processed objectPoints from the first viewpoint
	      //  into inCal.
	      // If the number of boards is one, then inCal will have as many
	      //  objectPoints as in inCalList[n][0]
	      inCal.objectPoints.insert(inCal.objectPoints.end(),
					arCalList[n][0].objPoints.begin(),
					arCalList[n][0].objPoints.end());
	      inCal.imagePoints.insert(inCal.imagePoints.end(),
				       arCalList[n][0].imgPoints.begin(),
				       arCalList[n][0].imgPoints.end());
	  
	      if (s.mode == Settings::STEREO){
                      
		// inCal2 is the final structure used for stereo calibration.
		//  Thus, move all the processed objectPoints from the second viewpoint
		//  into inCal2.
		// If the number of boards is one, then inCal2 will have as many
		//  objectPoints as in inCalList[n][1]
		inCal2.objectPoints.insert(inCal2.objectPoints.end(),
					   arCalList[n][1].objPoints.begin(),
					   arCalList[n][1].objPoints.end());
		inCal2.imagePoints.insert(inCal2.imagePoints.end(),
					  arCalList[n][1].imgPoints.begin(),
					  arCalList[n][1].imgPoints.end());
	      }	    
	    }

	    runCalibrationAndSave(s, inCal, inCal2);	    	    
	    break;
	    
	  }

	  s.imageSize = img.size();
	  Mat imgCopy;
	  
	  for(int n = 0; n< s.numberOfBoards; n++)
	    {
	      currentArCal = &arCalList[n][currentView];
	      imgCopy = arucoDetect(s, img, *currentArCal, boardsList[n], letsBreak);
	      if (letsBreak) break;
	      if(save) {
		sprintf(imgSave, "%sdetected_%d.jpg", s.detectedPath.c_str(), i);
		imwrite(imgSave, imgCopy);
	      }
	    }

	  if (letsBreak)
	    {
	      destroyWindow("Detected");
	      break;
	    }
	}
    }
    
/*-----------Calibration using Standard Chessboard--------------*/ 
    else if(s.calibrationPattern == Settings::CHESSBOARD){
      
      namedWindow("Detected", CV_WINDOW_AUTOSIZE);
      // For each image in the image list
      for(int i = 0;;i++)
	{
	  // Switches between intrinsic calibration structs for stereo mode
	  if (i%2 == 0) {
	    currentInCal = &inCal;	   
	  } else if (s.mode == Settings::STEREO){	    
	    currentInCal = &inCal2;
	  }

	  
	  // Set up the image
	  Mat img = s.imageSetup(i);
	  
	  // If there is no data, the photos have run out
	  if(!img.data) {
	    destroyWindow("Detected");
	    runCalibrationAndSave(s, inCal, inCal2);  
	    break;
	  }
	  
	  s.imageSize = img.size();
	  //Detect the pattern in the image, adding data to the imagePoints
	  //and objectPoints calibration parameters
	  chessboardDetect(s, img, *currentInCal);

	  if(save) {
	    sprintf(imgSave, "%sdetected_%d.jpg", s.detectedPath.c_str(), i);
	    imwrite(imgSave, img);
	  } 
	  
	  imshow("Detected", img);
	  // If wait setting is true, wait till next key press (waitkey(0)).
	  //  Otherwise, wait 50 ms
	  char c = (char)waitKey(s.wait ? 0: 10000);
	  if( (c & 255) == 27 || c == 'q' || c == 'Q' ){
	    destroyWindow("Detected");
	    break;
	  }
	}
    }
    
    return 0;
}





