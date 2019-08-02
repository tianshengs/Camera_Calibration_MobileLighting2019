// Tiansheng Sun, Middlebury College, Summer 2019
//  Creates a ChArUco chessboard. It is possible to specified the (numerically) first marker id on the board.
//  Based the first id, the board is drawn using it and the subsequent ids.
//  For example, a 4x4 board with first marker 1, uses the following ids: 1,2,3,...,25.
#include <stdio.h>
#include <time.h>
#include <vector>
#include "opencv2/aruco/dictionary.hpp"
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>


using namespace cv;
std::vector< std::vector< Point3f >> obj_points_vector;
std::vector< int > ids_vector;

namespace {
const char* about = "Create a ChArUco board image";
const char* keys  =
        "{@outfile |<none> | Output image }"
        "{w        |       | Number of chessboard squares in X direction }"
        "{h        |       | Number of chessboard squares in Y direction }"
        "{l        |       | Marker side length (in points) }"
        "{s        |       | Chessboard square side length (in points) }"
        "{d        | 11    | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, \n"
        "                  DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, \n"
        "                  DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, \n"
        "                  DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16} "
        "{b        | 1     |'1': save maps with border, '0': no border}"
        "{bb       | 1     | Number of bits in marker borders }"
        "{f        |       | first marker: range from 1 to [dictionary size] }"
        "{si       | false | show generated image }";
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

//the main function
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 7) {
        parser.printMessage();
        return 0;
    }

    Mat boardImage;

    //define the variables used to create ChArUco pattern
    int squareX = parser.get<int>("w");
    int squareY = parser.get<int>("h");
    float squareLength = parser.get<int>("s");
    float markerLength = parser.get<int>("l");
    int dictionaryId = parser.get<int>("d");
    int borderBits = parser.get<int>("bb");
    int border = parser.get<int>("b");
    bool showImage = parser.get<bool>("si");
    int firstMarker  = parser.get<int>("f");

    String out = parser.get<String>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    //get dictionary
    Ptr<aruco::Dictionary> dictionary =
      aruco::getPredefinedDictionary
      (aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    //create and save a new Charuco board
    Ptr<aruco::CharucoBoard> board = generateCharucoBoard(squareX, squareY, squareLength,
                                                         markerLength, dictionary, firstMarker);

    //define the image size
    Size imageSize;

    if (border) {
        imageSize.width = squareX * squareLength + 2 * borderBits;
        imageSize.height = squareY * squareLength + 2 * borderBits;
    } else{
        imageSize.width = squareX * squareLength;
        imageSize.height = squareY * squareLength;
    }

    //draw the image
    board->draw(imageSize, boardImage, borderBits, 1);

    //if showImage, show the image
    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(out, boardImage);

    waitKey(0);

    return 0;
}
