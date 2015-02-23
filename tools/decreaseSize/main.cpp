/**
 * @file main.cpp
 * @brief Decrease the quality of a video
 * @author Fabien ROUALDES and HE Huilong
 * @date 19/08/2013
 */
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "cv.h"

int main(int argc, char* argv[]){
  for(int f=1; f<argc; f++){
  std::string vInput(argv[f]);
  std::string vOutput(vInput + "out.avi");
 
  std::cout << "======" << std::endl; 
  std::cout << "INPUT: " << vInput << std::endl;
  cv::VideoCapture capture(vInput);

  double fpsIn = capture.get(CV_CAP_PROP_FPS);
  double fpsOut = fpsIn;
  /*
  double ratio = fpsOut/fpsIn; 
  std::cout << "Ratio fpsIn/fpsOut=" << ratio << std::endl;
  if(ratio > 1){
    std::cerr << "You cannot have a fps output superior of the original video!" << std::endl;
    return EXIT_FAILURE;
  }
  */ 
  // Change this size:
  cv::Size fsize(160,120);

  int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);

  std::cout << "* Number of frames: " << frameCount << std::endl;
  std::cout << "* Frames per second: " << fpsIn << std::endl;

  int fourcc = capture.get(CV_CAP_PROP_FOURCC);
  cv::VideoWriter writer(vOutput,
			 fourcc,
			 fpsOut,
			 fsize); 
  if(!writer.isOpened()){
    std::cerr << "Error while opening Writer!" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat mat;
  cv::Mat mConverted(fsize.height,fsize.width, CV_8UC1);
  int maxFrames = frameCount;//ratio*frameCount;

  // Selecting the frames
  int *values = new int[maxFrames];
  for(int i=0 ; i<maxFrames ; i++){
    values[i] = i;
  }
/*   
  if(fpsIn*2 > frameCount) {
    std::cerr << "Unexpected frame number. Aborting" <<std::endl;
    exit(EXIT_FAILURE);
  }
*/
  for(int i=0 ; i<maxFrames ; i++){
    capture.set(CV_CAP_PROP_POS_FRAMES, (double) values[i]);
    capture >> mat;
    resize(mat,mConverted,fsize);
    writer << mConverted;
  }
  
  //remove(vInput.c_str());    
  std::cout << "OUTPUT: " << vOutput << std::endl;
  std::cout << "* Number of frames: " << maxFrames << std::endl;
  std::cout << "* Frames per second: " << fpsOut << std::endl;
  delete [] values;
  } 
  return EXIT_SUCCESS;
}
