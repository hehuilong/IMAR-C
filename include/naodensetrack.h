#ifndef DENSETRACK_H_
#define DENSETRACK_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <unistd.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "KMdata.h"

#include "IplImageWrapper.h"
#include "IplImagePyramid.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef struct TrackerInfo{
  int trackLength; // length of the trajectory
  int initGap; // initial gap for feature detection
} TrackerInfo;

typedef struct DescInfo{
  int nBins; // number of bins for vector quantization
  int fullOrientation; // 0: 180 degree; 1: 360 degree
  int norm; // 1: L1 normalization; 2: L2 normalization
  float threshold; //threshold for normalization
  int flagThre; // whether thresholding or not
  int nxCells; // number of cells in x direction
  int nyCells; 
  int ntCells;
  int dim; // dimension of the descriptor
  int blockHeight; // size of the block for computing the descriptor
  int blockWidth;
} DescInfo; 

typedef struct DescMat
{
  int height;
  int width;
  int nBins;
  float* desc;
}DescMat;

class PointDesc
{
 public:
  std::vector<float> hog;
  std::vector<float> hof;
  std::vector<float> mbhX;
  std::vector<float> mbhY;
  CvPoint2D32f point;

 PointDesc(const DescInfo& hogInfo, const DescInfo& hofInfo, const DescInfo& mbhInfo, const CvPoint2D32f& point_)
   : hog(hogInfo.nxCells * hogInfo.nyCells * hogInfo.nBins),
    hof(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
    mbhX(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
    mbhY(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
    point(point_)
    {}
};

class Track
{
 public:
  std::list<PointDesc> pointDescs;
  int maxNPoints;
  
 Track(int maxNPoints_)
   : maxNPoints(maxNPoints_)
  {}
  
  void addPointDesc(const PointDesc& point)
  {
    pointDescs.push_back(point);
    if (pointDescs.size() > (unsigned int)maxNPoints + 2) {
      pointDescs.pop_front();
    }
  }
};

/* Descriptors */
/* get the rectangle for computing the descriptor */
CvScalar getRect(const CvPoint2D32f point, // the interest point position
		 const CvSize size, // the size of the image
		 const DescInfo descInfo); // parameters about the descriptor
/* compute integral histograms for the whole image */
void BuildDescMat(const IplImage* xComp, // x gradient component
		  const IplImage* yComp, // y gradient component
		  DescMat* descMat, // output integral histograms
		  const DescInfo descInfo); // parameters about the descriptor
/* get a descriptor from the integral histogram */
std::vector<float> getDesc(const DescMat* descMat, // input integral histogram
			   CvScalar rect, // rectangle area for the descriptor
			   DescInfo descInfo, // parameters about the descriptor
			   float epsilon);

void HogComp(IplImage* img, DescMat* descMat, DescInfo descInfo);
void HofComp(IplImage* flow, DescMat* descMat, DescInfo descInfo);
void MbhComp(IplImage* flow, DescMat* descMatX, DescMat* descMatY, DescInfo descInfo);

/* tracking interest points by median filtering in the optical field */
void OpticalFlowTracker(IplImage* flow, // the optical field
			std::vector<CvPoint2D32f>& points_in, // input interest point positions
			std::vector<CvPoint2D32f>& points_out, // output interest point positions
			std::vector<int>& status); // status for successfully tracked or not
/* check whether a trajectory is valid or not */
int isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length,
	    float min_var, float max_var, float max_dis);
/* detect new feature points in the whole image */
void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points,
		   const double quality, const double min_distance);
/* detect new feature points in a image without overlapping to previous points */
void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,
		   std::vector<CvPoint2D32f>& points_out, const double quality, const double min_distance);
// initialize
void InitTrackerInfo(TrackerInfo* tracker, int track_length, int init_gap);
DescMat* InitDescMat(int height, int width, int nBins);
void ReleDescMat( DescMat* descMat);
void InitDescInfo(DescInfo* descInfo, int nBins, int flag, int orientation, int size, int nxy_cell, int nt_cell, float min_flow);
void usage();
//void arg_parse(int argc, char** argv);
//int extractHOGHOF(std::string video, int dim, int maxPts, KMdata* dataPts);
//int extractMBH(std::string video, int dim, int maxPts, KMdata* dataPts);
int extract_feature_points(std::string video,
			   int scale_num,
			   std::string descriptor,
			   int dim,
			   int maxPts,
			   KMdata& dataPts);

#endif /*DENSETRACK_H_*/
