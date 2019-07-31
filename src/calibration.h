#ifndef _calibration_H
#define _calibration_H
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/aruco/charuco.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdio.h>
#include <string>
#include <time.h>
//#include "aruco.h"

using namespace std;
using namespace cv;
using namespace aruco;

//------------------Struct to store parameters for intrinsic calibration------------------//
struct intrinsicCalibration {
  Mat cameraMatrix, distCoeffs;   //intrinsic camera matrices
  vector<Mat> rvecs, tvecs;       //extrinsic rotation and translation vectors for each image
  vector<vector<Point2f> > imagePoints;   //corner points on 2d image
  vector<vector<Point3f> > objectPoints;  //corresponding 3d object points
  vector<float> reprojErrs;   //vector of reprojection errors for each pixel
  double totalAvgErr = 0;     //average error across every pixel
  vector< vector< vector< Point2f > > > allCorners;
  vector< vector< int > > allIds;

};

//------------------Struct to store parameters for stereo calibration------------------//
struct stereoCalibration {
    Mat R, T, E, F;         //Extrinsic matrices (rotation, translation, essential, fundamental)
    Mat R1, R2, P1, P2, Q;  //Rectification parameters ()
    Rect validRoi[2];       //Rectangle within the rectified image that contains all valid points
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
    Settings();

    enum Pattern { CHESSBOARD, ARUCO_SINGLE, CHARUCO, NOT_EXISTING };
    enum Mode { INTRINSIC, STEREO, INVALID };

    void write(FileStorage& fs) const;
    void read(const FileNode&);
    void interprate();
    cv::Mat imageSetup(int imageIndex);

    bool readImageList(const string& filename);
    bool readIntrinsicInput(const string& filename);
    void saveIntrinsics(intrinsicCalibration &inCal);
    void saveExtrinsics(stereoCalibration &sterCal);

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
  vector<float> squareLength;
  vector<float> markerLength;  // The length of the aruco markers in pixels
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

bool pathCheck(const string& path);
void printMat(Mat m, const char *name);
void printPoints(const intrinsicCalibration inCal);
void getSharedPoints(intrinsicCalibration &inCal, intrinsicCalibration &inCal2);
void getObjectAndImagePoints( vector< vector< Point2f > >  detectedCorners, vector< int > detectedIds,
			      vector< Point3f > &objPoints, vector< Point2f > &imgPoints,
			      Ptr<ChessBoard> &currentBoard);
void processPoints(Settings s, vector< vector< Point2f > > corners, vector<int> ids,
		   vector<int> counter, vector< vector < Point2f >> &processedImagePoints,
		   vector< vector<Point3f>> &processedObjectPoints, Ptr<ChessBoard> &currentBoard);
void setUpAruco( Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2, Ptr<ChessBoard> &currentBoard);
void setUpAruco_( Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2, Ptr<ChessBoard> &currentBoard, int n);
void arucoDetect(Settings s, Mat &img, intrinsicCalibration &InCal, Ptr<ChessBoard> currentBoard);
void runCalibrationAndSave(Settings s, intrinsicCalibration &inCal, intrinsicCalibration &inCal2);
void chessboardDetect(Settings s, Mat &img, intrinsicCalibration &inCal);
int calibrateWithSettings( const string inputSettingsFile );
vector<int> detectionCheck( char* settingsFile, char* filename0, char* filename1 = NULL);
