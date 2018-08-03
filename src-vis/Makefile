SRC =  calibration.cpp  calibrateWithSettings.cpp 

BIN = calibrateWithSettings   \
      imdiff create_chessboard_obj imageList_creator \
      create_new_chessboard detection_check det_check

DBG= -g
CC = g++
WARN= -W -Wall 
CPPFLAGS = -O2 $(WARN) $(DBG) -I$(IMGLIB) -I/usr/include/opencv2
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d \
	-lopencv_features2d -lopencv_imgcodecs -lopencv_videoio -lopencv_aruco


OBJ = $(SRC:.cpp=.o)

all: $(BIN)

calibrateWithSettings: $(OBJ)
	$(CC) -o $@  $(OBJ) $(LDLIBS)

clean:
	rm -f $(BIN) *.o core*
