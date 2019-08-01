BIN = imdiff imageList_creator create_new_chessboard ChArCo


DBG= -g
CC = g++
WARN= -W -Wall 
CPPFLAGS = -O2 $(WARN) $(DBG) -I$(IMGLIB) -I/usr/include/opencv2
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d \
	-lopencv_features2d -lopencv_imgcodecs -lopencv_videoio -lopencv_aruco


all: $(BIN)

clean: rm -f $(BIN) *.o core*
