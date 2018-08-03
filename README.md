# Camera Calibration Summer 2018

A C++ camera calibration program that performs both intrinsic and stereo
calibration. It supports calibration with more than one board in the scene.
Furthermore, it provides information about the region of interest of the image pairs
after rectification. 

These programs were developed for a Middlebury College undergraduate summer research project,
led by professor Daniel Scharstein. Camera calibration will be one component of
a pipeline designed to create datasets for 3D reconstruction on mobile devices.
These datasets will become the next entry in the well known Middlebury Computer Vision
Benchmark http://vision.middlebury.edu/. We presented at Middlebury Summer Research Symposium; [our poster](misc/readme/poster.pdf)
from our summer research presentation.

## Overview
There are essentially two versions of the main program ([*calibration*](src-vis/calibration.cpp)), which have slightly different features.
Both versions perform two types of camera calibration (intrinsics and extrinsics).
Both versions use the OpenCV AruCo moduled. They are controlled by a settings class, which is inputted as a YAML file.
These settings control input and output features, such as  the ability to save detected,
undistorted, and rectified images, and the ability to change the calibration type.

### Calibration w/ visualizations.
This program can be found in [src-vis](src-vis).
This version provides visual feedback to the user:
* It pops up a "Detection" window, showing the detection of the calibration patterns. Within the detection window, press `c` to move from a picture to the next, which stands for *consider*.
  If multiple calibration patterns are used simultaneously. Press `c` as many times as the number of calibration pattern on the scene. Press `Esc` or `q` to quit.
* It pops up a "Rectification" window, showing a stereo image pair after rectification. Again, press `c` to move from a picture to the next.

### Calibration w/o visualizations.
This program can be found in src.
This version does not provide visual feedback, and it simply outputs the intrisic and extrinsic parameters.

### Additional remarks.
There are other ausiliary programs in each src folder (and src-vis). All the programs in src are designed to be implemented within larger applications such as Middlebury College [MobileLighting](https://github.com/nmosier/MobileLighting). I advice to use the files in src-vis when "stand-alone" calibration functionalities are needed.[1]

## Installation
### Dependencies
This program requires at least [OpenCV3*](http://opencv.org/releases.html). Since OpenCV3 release, the ArUco library
is a built in module. All program were developed in OpenCV 3.2, but they will work with all recent OpenCV versions. More on OpenCV compatibility in the section, **Changes Between OpenCV3.2 and OpenCV3.3**. 
Extensive step-by-step guides about installation can be found on the [OpenCV Tutorials](https://docs.opencv.org/3.2.0/df/d65/tutorial_table_of_content_introduction.html) webpage.

## Usage

### Compiling
Each folder contains a folder-specific *Make* file. Then, it is easy task to compile all necessary programs using the command line: 
`Make`

### Getting started

The calibration program is run from settings files, which are YAML or XML (this functionality is
adapted from the [2017 MobileLighting Research Project](https://github.com/kylebmeredith/Camera-Calibration).
The [settings directory](misc\input\settings) includes two example settings files, which detailed documentation for each entry.
When the settings file and the main program are in the same folder, here is a sample program execution command: `./calibrateWithSettings ./intrinsicSettings.yml`

The calibration program can also write a serialization for settings, using the settings class function write().
To use this functionality, you must uncomment the other write() function outside of the settings class
(check out the [OpenCV Filestorage documentation](http://docs.opencv.org/3.0-rc1/dd/d74/tutorial_file_input_output_with_xml_yml.html) for more information).

Two modes are supported: **INTRINSIC**, **STEREO**.  
Two calibration patterns are supported: **CHESSBOARD**, **ARUCO_SINGLE**, where CHESSBOARD represents a traditional black-and-white chessboard.

All modes require a YAML/XML [image list](misc\input\imagelists) with paths to the input
[images](misc\input\imagelists), specified by the setting: **imageList_Filename**.

### ArUco Calibration Patterns
ArUco patterns are barcode patterns, and they are comprised of markers with unique IDs based on a modified Hamming code. 

The number of unique IDs depends on the predefined AruCo dictionary. A marker ID is the marker index inside the dictionary it belongs to. For example, the first 3 markers inside a dictionary have IDs: 0, 1, and 2. Note that each dictionary is composed by a different *type* and *number* of markers. The default dictionary for the calibration programs is number 11, i.e. DICT_6X6_1000. Concretely, this dictionary is composed by 1000 unique markers and a marker size of 6x6 bits (DICT_6X6_1000). Markers IDs are important: they are used during detection to check whether a detected square is actually an AruCo marker of interest. 

When creating a new AruCo chessboard (using *create_new_chessboard.cpp*), the user will be promped to enter the "First ID" of the board. For example, if the user chooses the first ID to be 300, and the AruCo chessboard contains 100
markers, then the board will have IDs: 300, 301, 302, ... 399. It is important to remember or record the first ID of each board you create. The first ID functions like the *board identifier*, and it is required in the settings file to run the main calibration program:
```
 #First marker must be between 0 and the dictionary size. For two or more boards, give data as an array.
  First_Marker: [0, 113]
```

To create your own ArUco chessboard, use the [*create_new_chessboard*](utils/createArucoPatterns.cpp)
utility program. This program will create a new chessboard with unique markers, allowing simultaneous detection of multiple patterns.
In a terminal, type: 
`./create_new_chessboard -h` 
for more information about all the possible input parameters.

### Intrinsic Calibration     
Intrinsic mode uses OpenCV's [calibrateCamera function](https://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) to perform intrinsic camera calibration. It requires an imageList with
images from a single viewpoint ([example set](misc\input\images\intrinsics\)). It can be run
with  both calibration patterns. The first step of the calibration pipelone is to calculate
camera intrinsics using a high quality set of (Aruco or traditional) chessboard images. The images must be collected in a .yml image list. In order to compute intrinsic camera parameters successfully, it is advices to do the following:
* Take pictures of calibration pattern such that it appears in the corners *and* on the edges of the camera field of view.
* At each selected position, take pictures of the calibration pattern with various angles: tilted to the left, tilted to the right, tilted backwards. This is necessary for accurate focal lengths estimation.
* Take about 15 pictures. However, try to make each pictures as different as possible from the others -- duplicates add more noise than actual useful data points.

Intrinsic calibration can be optimized by modifying the flags in the calibrateCamera function
(check the API linked above for more information). The setting **Calibrate_FixDistCoeffs**
is a string of five digits (0 or 1) that controls which distortion coefficients among K1-K5
will be fixed at 0 (1 = fixed). It is suggested to fix distortion coefficients 4 and 5 ("00011").
The third radial distortion coeffienct accouts for complex distortion such as "mustache distortion", and reduces the amount of *noise* in the final rectfied images.
For more information about distortion corrections can be found on the [OpenCV](https://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html) website.
From the settings file, additional flags can be added: CV_CALIB_FIX_PRINCIPAL_POINT, CV_CALIB_FIX_ASPECT_RATIO,
and CV_CALIB_ZERO_TANGENT_DIST.

The program will output the resulting intrinsics in a file specified by the setting:
**IntrinsicOutput_Filename**. The file will contain the calibration configuration (time, pattern, and flags),
and the calibration results (camera matrix, distortion coefficients, and reprojection error).
These intrinsic files can be used as intrinsic input for future calibration, using the setting: **intrinsicInput_Filename**.

Intrinsic parameters can also be used to correct the radial distortion in the input
images. The setting **Show_UndistortedImages** controls whether or not these undistorted images
are shown after calibration. If the setting **UndistortedImages_Path** is changed from "0,"
the program will try to save these images to the path. It will print an error if this path does
not exist (*the path must be created beforehand*).

### Stereo Calibration
Stereo mode uses OpenCV's [stereoCalibrate function](https://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#ga246253dcc6de2e0376c599e7d692303a)
to perform extrinsic calibration. It requires an imageList with image pairs of an
identical scene from two viewpoints ([example set](misc\input\images\extrinsics)). The order
of image paths within the image list is important: it must alternate between viewpoints (left1 right1 left2 right2).
With the both calibration patterns, the intrinsic input is optional but strongly adviced for a good calibration. If it is left at "0," the program will calculate
independent intrinsics for each viewpoint and input these into the stereoCalibrate function.

Extrinsic calibration is the second step of the calibration pipeline, and to be achieved successfully I suggest the following:
* Capture images with two calibration boards in the scene. It is possible to use two boards by changing the settings file, **Num_of_Boards: 2**. Then, the other AruCo settings must be
specified for each board in the scene. For example:
```
The length of the AruCo markers in pixels
Marker_Length: [108, 72]
```
where one board would have marker length 1.5 inches (108 px) and the other board would have marker length 1 inch (72 px). 
* During data acquisition, "span" the depth of the scene with one board, while keeping the other closer to the camera.
This ensures the efficient acquisition of enough depth data, which is required for a successful computation of the scene depth maps.
Following this procedure leads to rubust and reliable rotation and translation matrices, which are thereby the output of extrinsic calibration.
* Take about 20 pictures of the calibration patterns from the two stereo viewpoints. It is a good rule of thumb for an accurate calibration pipeline.

The program plugs these resulting extrinsics into OpenCV's [stereoRectify function](https://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6),
which calculates the necessary rectification transformations and projection matrices
to create rectified image pairs. It is possible to specify some rectification settings in the settings file:
- The `Alpha_parameter` or free scaling factor. The default is `-1`.   # Alpha_parameter: the free scaling factor.
  If -1, the focal lengths of the camera are kept fixed during rectification -- when computing the projection matrices.
  If 1, the rectified images are decimated and shifted so that all the pixels from the original image are retained in the rectified image -- focal lengths get reduced in the process.
  If 0, the received pictures are zoomed and shifted so that only valid pixels are visible -- focal lengths get increased in the process.
- The `Resizing_factor`. It is an integer factor that controls the size of the rectified images relative to the pictures original size.
  It is useful when the user is planning to perform an accurate cropping of the rectified images. If the rectified images are 3 times as larger than the original size,
  then it will be extremelt easy to crop them accurantly. This connect to the last settings option, which follows.
- The `Cropping_After_Rectification`. If 1, the program computes the masks to crop the rectified images down to their region of interest. The masks are computed using OpenCV thresholding
  functions, and then connectedComponentsWithStats() which finds quicly and efficiently the region of interest of the rectified image.
  This allows the user to remove the (uncessary) black background, which often takes a large portion of the rectified images. 

Furthermore, the setting **Show_RectifiedImages** controls whether or not these rectified images are shown after calibration.
If the setting **RectifiedImages_Path** is changed from "0," the program will try to save these images to the path. It will
print an error if this path does not exist (*the path must be created beforehand*).

Using the utility program [imdiff](utils/imdiff.cpp), you can compare the rectified images
and check how well they are rectified. In a perfectly rectified image pair, each point on one image is on the same horizontal line as the correspoding point on the other image.

The program will output the resulting extrinsics in a file specified by the setting:
**ExtrinsicOutput_Filename**. The file will contain the calibration configuration (time and pattern); the original size of the images;
the stereo calibration paramaters (rotation matrix, translation vector, and essential/fundamental matrices); and
the rectification parameters (rectification transformations, projection matrices,
and disparity-to-depth mapping matrix). 
If `Cropping_After_Rectification` is 1, the program will output an additional extrinsics file, **ExtrinsicOutput_Filename_withMasks**,
which contains *also* the masks for cropping the rectified images down to the region of interest.

## Auxiliary Programs

### Feedback detection
The auxiliary C++ program [*detection_check*](src-vis/detection_check.cpp) can be used to provide general feedback regarding the status of the AruCo detection. There are two versions of the program.
* The version in *src-vis* is the stand-alone version. It takes the two images in a stereo pair and the *same* settings file used for calibration as arguments:
```
$ ./detection_check
  Usage: ./detection_check [path to settings file] [path to img1] [path to img2]
```
* The version in *src* is designed to be called by larger applications. It does not have a *main* method, and thereby it works like a simple script. Whem called within larger applications, it returns the number of corners detected for each board from the two stereo viewpoints, and it returns also the number of detected corners that are shared between viewpoints. 

### Image lists
The auxiliary C++ program [*imageList_creator*](src-vis/imageList_creator.cpp) can be used to create a new image list. If the order of the images in the image list is not relevant (e.g. during intrinsic calibration), you can create a new image list as follows:
```
./imageList_creator [imagelist_filename.yml] [path to images]
```
You can also create an image list with the necessary format for stereo calibration:
```
./imageList_creator [imagelist_filename.yml] [path to images] [-stereo=true]
```

### New chessboards patterns
The auxiliary C++ program [*create_new_chessboard*](src-vis/create_new_chessboard.cpp) was introduced in the "**ArUco Calibration Patterns**" paragraph. The output of `./create_new_chessboard -h` is below:
```
Usage: create_new_chessboard [params] outfile

        -b (value:1)
                '1': save maps with border, '0': no border
        --bb (value:1)
                Number of bits in marker borders
        -d (value:11)
                dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, 
                  DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7,  
                  DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,  
                  DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16  
        -f
                first marker: range from 1 to [ictionary size]
        -h
                Number of markers in Y direction
        -l
                Marker side length (in pixels)
        -s
                Separation between two consecutive markers in the grid (in pixels) -- should be same as marker length
        --si (value:false)
                show generated image
        -w
                Number of markers in X direction

        outfile (value:<none>)
                Output image (.png)
```

## Closing Remarks and Tips


##### Footnotes
[1] In fact, it is  possible to use almost all programs in the *src* directory even outside of larger applications.
However, *detection_check.cpp* (a detection feedback program) in the *src* directoty can only live within larger applications (because it does not posses a *main* method).