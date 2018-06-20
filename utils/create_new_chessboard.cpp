#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"
#include <time.h>
#include <vector>
#include "opencv2/aruco/dictionary.hpp"
#include <stdlib.h>


using namespace cv;
using namespace std;


std::vector< std::vector< Point3f >> obj_points_vector;
std::vector< int > ids_vector;


using namespace cv;

namespace {
const char* about = "Create an ArUco grid board image";
const char* keys  =
        "{@outfile |<none> | Output image }"
        "{w        |       | Number of markers in X direction }"
        "{h        |       | Number of markers in Y direction }"
        "{l        |       | Marker side length (in pixels) }"
        "{s        |       | Separation between two consecutive markers in the grid (in pixels)}"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{m        |       | Margins size (in pixels). Default is marker separation (-s) }"
        "{b        | 1     |'1': save maps with border, '0': no border (1 default)}"
        "{bb       | 1     | Number of bits in marker borders }"
        "{n        | 1     | Number of boards}"
        "{r        |       | Board type: range 1-dictionary }"
        "{si       | false | show generated image }";
}


class ChessBoard : public aruco::GridBoard {

public:
  void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);
  
  static Ptr<ChessBoard> create(int markersX, int markersY, float markerLength,
				float markerSeparation,
				const Ptr<aruco::Dictionary> &dictionary,
				int firstMarker = 0);

private:
  // number of markers in X and Y directions
  int __markersX, __markersY;
  
  // marker side lenght (normally in meters)
  float __markerLength;
  
  // separation between markers in the grid
  float __markerSeparation;
  
};


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

    //srand(firstMarker);
    
    // fill ids with first identifiers
    for(unsigned int i = 0; i < totalMarkers; i++) {
      res->ids[i] =  i + firstMarker;    //rand()%1000 ;
      cout << res->ids[i] << endl;
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

    
    return res;
}



int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 7) {
        parser.printMessage();
        return 0;
    }

    int markersX = parser.get<int>("w");
    int markersY = parser.get<int>("h");
    int markerLength = parser.get<int>("l");
    int markerSeparation = parser.get<int>("s");
    int dictionaryId = parser.get<int>("d");
    int margins = markerSeparation;
    if(parser.has("m")) {
        margins = parser.get<int>("m");
    }

    int borderBits = parser.get<int>("bb");
    int border = parser.get<int>("b");
    bool showImage = parser.get<bool>("si");
    //int numBoards  = parser.get<int>("n");
    int randomSeed  = parser.get<int>("r");
    

    String out = parser.get<String>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Size imageSize;
    imageSize.width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
    imageSize.height = markersY *(markerLength + markerSeparation) - markerSeparation + 2 * margins;

  
    
    Ptr<aruco::Dictionary> dictionary =
      aruco::getPredefinedDictionary
      (aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    
    
    

    Ptr<ChessBoard> board = ChessBoard::create(markersX, markersY, float(markerLength),
					       float(markerSeparation), dictionary, randomSeed);


    // show created board  
    Mat boardImage;
    drawPlanarBoard(board, imageSize, boardImage, margins, borderBits);


    
    if (border) {
      float maxY = (float) markersY * markerLength + (markersY - 1) * markerSeparation + margins ;
      float maxX = (float) markersX * markerLength + (markersX - 1) * markerSeparation + margins ;
      float bSize = (markerLength * 0.05);
      int borderX = 0;
      int borderY = 0;
      
      for (int i = 0; i < 2; i++) {	
	for(int x = 0; x <  markersX+1 ; x++) {
	    Point2f topLeft  = Point2f(x * (markerLength + markerSeparation), borderY);
	    Point2f bottomRight = topLeft + Point2f(markerLength-1, markerLength-1);
	    cv::rectangle(boardImage, topLeft, bottomRight, Scalar(0,0,0) , -1);
	}
	borderY = maxY;
      }
      
      for (int i = 0; i < 2; i++) {	
	for(int y = 0; y <  markersY+1 ; y++) {
	  Point2f topLeft  = Point2f(borderX, y * (markerLength + markerSeparation));
	  Point2f bottomRight = topLeft + Point2f(markerLength-1, markerLength-1);
	  cv::rectangle(boardImage, topLeft, bottomRight, Scalar(0,0,0) , -1);
	}
	borderX = maxX;
      }
      
      
	Mat cropped =
	  boardImage(Rect(markerLength * 0.5 , markerLength *0.5,
			  imageSize.width - markerLength ,
			  imageSize.height - markerLength));
      
	cropped.copyTo(boardImage);

    }
 

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(out, boardImage);
    

    return 0;
}
