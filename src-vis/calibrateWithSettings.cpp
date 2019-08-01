#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace std;
using namespace cv;

#include "calibration.h"

int main( int argc, char** argv )
{
    const char * inputSettingsFile;

    //for calibration
    if (argc == 2) {
        inputSettingsFile = argv[1];
        calibrateWithSettings(inputSettingsFile);
    }

    //for one board detection check
    else if (argc == 3) {
        detectionCheck(argv[1], argv[2]);
    }

    //for stereo detection check
    else if (argc == 4) {
        detectionCheck(argv[1], argv[2], argv[3]);
    }

    else {
        cerr << "Usage: calibrateWithSettings [path to settings file]"
             << "to do calibration;" << endl << endl
             << "Usage: calibrationWithSettings [path to settings file] [img0] [img1] "
             << "to do detection check." << endl << endl
             << "The settings folder contains several example files with "
                "descriptions of each parameter. Check the README for more detail." << endl;
        return -1;
    };

};
