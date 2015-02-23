/**
 * @file main.cpp
 * @brief Decrease the quality of a video
 * @author Fabien ROUALDES
 * @date 19/08/2013
 */
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "cv.h"

int main(int argc, char* argv[]){
  if(argc < 2){
    std::cerr << "usage: detfps filename"<<std::endl;
    exit(EXIT_FAILURE);
  }
  for(int i=1;i<=argc-1;i++){
    std::string filename(argv[i]);
    std::string vInput(filename);
    
    cv::VideoCapture capture(vInput);
    double fpsIn = capture.get(CV_CAP_PROP_FPS);

    int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
    std::cout << "INPUT: " << vInput << std::endl;
    std::cout << "* Number of frames: " << frameCount << std::endl;
    std::cout << "* Frames per second: " << fpsIn << std::endl;
  }
  return EXIT_SUCCESS;
}
