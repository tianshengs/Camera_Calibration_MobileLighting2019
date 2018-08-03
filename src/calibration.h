#ifndef _calibration_H
#define _calibration_H
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

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

int calibrateWithSettings( const string inputSettingsFile );


struct intrinsicCalibration {
  Mat cameraMatrix, distCoeffs;   //intrinsic camera matrices
  vector<Mat> rvecs, tvecs;       //extrinsic rotation and translation vectors for each image
  vector<vector<Point2f> > imagePoints;   //corner points on 2d image
  vector<vector<Point3f> > objectPoints;  //corresponding 3d object points
  vector<float> reprojErrs;   //vector of reprojection errors for each pixel
  double totalAvgErr = 0;     //average error across every pixel
};

struct stereoCalibration {
    Mat R, T, E, F;         //Extrinsic matrices (rotation, translation, essential, fundamental)
    Mat R1, R2, P1, P2, Q;  //Rectification parameters ()
    Rect validRoi[2];       //Rectangle within the rectified image that contains all valid points
};

struct arucoCalibration {
    vector<vector<Point2f> > imgPoints;   //corner points on 2d image 
    vector<vector<Point3f> > objPoints;  //corresponding 3d object points prepared for calibration.
    
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
    Settings();

    enum Pattern { CHESSBOARD, ARUCO_SINGLE, NOT_EXISTING };
    enum Mode { INTRINSIC, STEREO, INVALID };

    void write(FileStorage& fs) const;
    void read(const FileNode& node);
    void interprate();
    cv::Mat imageSetup(int imageIndex);

    static bool readStringList(const string& filename, vector<string>& l);
    //static bool readConfigList( const string& filename, vector<MarkerMap>& l );
    static bool readIntrinsicInput( const string& filename, intrinsicCalibration& intrinsicInput );
    static bool fileCheck( const string& filename, FileStorage fs, const char * var);
    void saveIntrinsics(intrinsicCalibration &inCal);
    void saveExtrinsics(stereoCalibration &sterCal);
    
    
    // Properties
    Mode mode;
    Pattern calibrationPattern;   // One of the Chessboard, circles, or asymmetric circle pattern


    //AruCo configuration
    vector<int>  markersX;       // Number of AruCo Markers in first row
    vector<int>  markersY;       // Number of AruCo Markers in first column 
    vector<float> markerLength;  // The length of the markers in pixels
    int dictionary;              // The number of the AruCo dictionary used to draw the markers
    vector< int > type;          // The id of the first marker of the board
    int numberOfBoards;          // Number of boards in the scene. Default:1

    // Chessboard settings
    Size boardSize;                 // The size of the board -> Number of items by width and height
    float squareSize;               // The size of a square in your defined unit (point, millimeter,etc)
    
    // Input settings
    vector<string> imageList;
    string imageListFilename;

    intrinsicCalibration intrinsicInput; // Struct to store inputted intrinsics
    string intrinsicInputFilename;    // Leave it at 0 to calculate new intrinsics
    bool useIntrinsicInput;

    // Output settings
    string intrinsicOutput;    // File to write results of intrinsic calibration
    string extrinsicOutput;    // File to write extrisics of stereo calibration

    string undistortedPath;    // Path at which to save undistorted images (leave 0 to not save undistorted)
    string rectifiedPath;      // Path at which to save rectified images (leave 0 to not save rectified)
    string detectedPath;       // Path at which to save images with detected patterns

    // Itrinsic calibration settings
    string fixDistCoeffs;              // A sequence of five digits (0 or 1) that
                                       //control which distortion coefficients will be fixed
    float aspectRatio;              // The aspect ratio. If it is inputted as 1, it will be fixed in calibration
    bool assumeZeroTangentDist;     // Assume zero tangential distortion
    bool fixPrincipalPoint;         // Fix the principal point at the center
    int flag;                       // Flag to modify calibration

    //Stereo Calibration settings
    int alpha; // Free scaling factor during rectification
    int rf;    // Resizing factor after rectification:
               //  decides how large the resolution of the
               //  image after rectification with respect to the original
               //  image size.
    
    // LEAVE THIS VALUE AT "0" TO NOT CROP RECTIFIED IMAGES
    int crop;  // Crops the final rectified images
    
    // UI settings
    bool showUndistorted;         // Show undistorted images after intrinsic calibration
    bool showRectified;           // Show rectified images after stereo calibration
    bool wait;                    // Wait until a key is pressed to show the next detected image

    // Program variables
    int nImages;
    Size imageSize;
    int nBoards;

    bool goodInput;

private:
    string modeInput;
    string patternInput;
    string cameraIDInput;
};

// static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings());
// static void write(FileStorage& fs, const string&, const Settings& x);
