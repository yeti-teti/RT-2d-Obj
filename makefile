CXX = g++
CXXFLAGS = -std=c++11
LIBS = -L/opt/homebrew/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_videoio -lopencv_videostab -lopencv_objdetect -lopencv_dnn
INCLUDES = -I/opt/homebrew/include/opencv4


objectRecognition: objectRecognition.cpp csv_util.cpp pipeline.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -f objectRecognition
