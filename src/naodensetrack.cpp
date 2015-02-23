/** 
 * \file naodensetrack.cpp
 * \brief  Set of function permiting to extract dense points and their trajectories.
 * \author LEAR
 * \author Fabien ROUALDES
 * \author Huilong HE
 * \date 19/08/2013
 */
#include "naodensetrack.h"

/* Descriptors */
/* get the rectangle for computing the descriptor */
CvScalar getRect(const CvPoint2D32f point, // the interest point position
		 const CvSize size, // the size of the image
		 const DescInfo descInfo) // parameters about the descriptor
{
  int x_min = descInfo.blockWidth/2;
  int y_min = descInfo.blockHeight/2;
  int x_max = size.width - descInfo.blockWidth;
  int y_max = size.height - descInfo.blockHeight;
  
  CvPoint2D32f point_temp;
  
  float temp = point.x - x_min;
  point_temp.x = std::min<float>(std::max<float>(temp, 0.), x_max);
  
  temp = point.y - y_min;
  point_temp.y = std::min<float>(std::max<float>(temp, 0.), y_max);
  
  // return the rectangle
  CvScalar rect;
  rect.val[0] = point_temp.x;
  rect.val[1] = point_temp.y;
  rect.val[2] = descInfo.blockWidth;
  rect.val[3] = descInfo.blockHeight;
  
  return rect;
}

/* compute integral histograms for the whole image */
void BuildDescMat(const IplImage* xComp, // x gradient component
		  const IplImage* yComp, // y gradient component
		  DescMat* descMat, // output integral histograms
		  const DescInfo descInfo) // parameters about the descriptor
{
  // whether use full orientation or not
  float fullAngle = descInfo.fullOrientation ? 360 : 180;
  // one additional bin for hof
  int nBins = descInfo.flagThre ? descInfo.nBins-1 : descInfo.nBins;
  // angle stride for quantization
  float angleBase = fullAngle/float(nBins);
  int width = descMat->width;
  int height = descMat->height;
  int histDim = descMat->nBins;
  int index = 0;
  for(int i = 0; i < height; i++) {
    const float* xcomp = (const float*)(xComp->imageData + xComp->widthStep*i);
    const float* ycomp = (const float*)(yComp->imageData + yComp->widthStep*i);
    
    // the histogram accumulated in the current line
    std::vector<float> sum(histDim);
    for(int j = 0; j < width; j++, index++) {
      float shiftX = xcomp[j];
      float shiftY = ycomp[j];
      float magnitude0 = sqrt(shiftX*shiftX+shiftY*shiftY);
      float magnitude1 = magnitude0;
      int bin0, bin1;
      
      // for the zero bin of hof
      if(descInfo.flagThre == 1 && magnitude0 <= descInfo.threshold) {
	bin0 = nBins; // the zero bin is the last one
	magnitude0 = 1.0;
	bin1 = 0;
	magnitude1 = 0;
      }
      else {
	float orientation = cvFastArctan(shiftY, shiftX);
	if(orientation > fullAngle)
	  orientation -= fullAngle;
		
	// split the magnitude to two adjacent bins
	float fbin = orientation/angleBase;
	bin0 = cvFloor(fbin);
	float weight0 = 1 - (fbin - bin0);
	float weight1 = 1 - weight0;
	bin0 %= nBins;
	bin1 = (bin0+1)%nBins;
		
	magnitude0 *= weight0;
	magnitude1 *= weight1;
      }

      sum[bin0] += magnitude0;
      sum[bin1] += magnitude1;
      
      int temp0 = index*descMat->nBins;
      if(i == 0) { // for the first line
	for(int m = 0; m < descMat->nBins; m++)
	  descMat->desc[temp0++] = sum[m];
      }
      else {
	int temp1 = (index - width)*descMat->nBins;
	for(int m = 0; m < descMat->nBins; m++)
	  descMat->desc[temp0++] = descMat->desc[temp1++]+sum[m];
      }
    }
  }
}

/* get a descriptor from the integral histogram */
std::vector<float> getDesc(const DescMat* descMat, // input integral histogram
			   CvScalar rect, // rectangle area for the descriptor
			   DescInfo descInfo, // parameters about the descriptor
			   float epsilon){
  int descDim = descInfo.dim;
  int height = descMat->height;
  int width = descMat->width;
  
  boost::numeric::ublas::vector<double> vec(descDim);
  int xOffset = rect.val[0];
  int yOffset = rect.val[1];
  int xStride = rect.val[2]/descInfo.nxCells;
  int yStride = rect.val[3]/descInfo.nyCells;
  
  // iterate over different cells
  int iDesc = 0;
  for (int iX = 0; iX < descInfo.nxCells; ++iX)
    for (int iY = 0; iY < descInfo.nyCells; ++iY) {
      // get the positions of the rectangle
      int left = xOffset + iX*xStride - 1;
      int right = std::min<int>(left + xStride, width-1);
      int top = yOffset + iY*yStride - 1;
      int bottom = std::min<int>(top + yStride, height-1);
      
      // get the index in the integral histogram
      int TopLeft = (top*width+left)*descInfo.nBins;
      int TopRight = (top*width+right)*descInfo.nBins;
      int BottomLeft = (bottom*width+left)*descInfo.nBins;
      int BottomRight = (bottom*width+right)*descInfo.nBins;
      
      for (int i = 0; i < descInfo.nBins; ++i, ++iDesc) {
	double sumTopLeft(0), sumTopRight(0), sumBottomLeft(0), sumBottomRight(0);
	if (top >= 0) {
	  if (left >= 0)
	    sumTopLeft = descMat->desc[TopLeft+i];
	  if (right >= 0)
	    sumTopRight = descMat->desc[TopRight+i];
	}
	if (bottom >= 0) {
	  if (left >= 0)
	    sumBottomLeft = descMat->desc[BottomLeft+i];
	  if (right >= 0)
	    sumBottomRight = descMat->desc[BottomRight+i];
	}
	float temp = sumBottomRight + sumTopLeft
	  - sumBottomLeft - sumTopRight;
	vec[iDesc] = std::max<float>(temp, 0) + epsilon;
      }
    }
  if (descInfo.norm == 1) // L1 normalization
    vec *= 1 / boost::numeric::ublas::norm_1(vec);
  else // L2 normalization
    vec *= 1 / boost::numeric::ublas::norm_2(vec);
  
  std::vector<float> desc(descDim);
  for (int i = 0; i < descDim; i++)
    desc[i] = vec[i];
  return desc;
}
 
void HogComp(IplImage* img, DescMat* descMat, DescInfo descInfo){
  int width = descMat->width;
  int height = descMat->height;
  IplImage* imgX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* imgY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  cvSobel(img, imgX, 1, 0, 1);
  cvSobel(img, imgY, 0, 1, 1);
  BuildDescMat(imgX, imgY, descMat, descInfo);
  cvReleaseImage(&imgX);
  cvReleaseImage(&imgY);
}

void HofComp(IplImage* flow, DescMat* descMat, DescInfo descInfo){
  int width = descMat->width;
  int height = descMat->height;
  IplImage* xComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
  IplImage* yComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
  for(int i = 0; i < height; i++) {
    const float* f = (const float*)(flow->imageData + flow->widthStep*i);
    float* xf = (float*)(xComp->imageData + xComp->widthStep*i);
    float* yf = (float*)(yComp->imageData + yComp->widthStep*i);
    for(int j = 0; j < width; j++) {
      xf[j] = f[2*j];
      yf[j] = f[2*j+1];
    }
  }
  BuildDescMat(xComp, yComp, descMat, descInfo);
  cvReleaseImage(&xComp);
  cvReleaseImage(&yComp);
}

void MbhComp(IplImage* flow, DescMat* descMatX, DescMat* descMatY, DescInfo descInfo){
  int width = descMatX->width;
  int height = descMatX->height;

  IplImage* flowX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* flowY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* flowXdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* flowXdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* flowYdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
  IplImage* flowYdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);

  // extract the x and y components of the flow
  for(int i = 0; i < height; i++) {
    const float* f = (const float*)(flow->imageData + flow->widthStep*i);
    float* fX = (float*)(flowX->imageData + flowX->widthStep*i);
    float* fY = (float*)(flowY->imageData + flowY->widthStep*i);
    for(int j = 0; j < width; j++) {
      fX[j] = 100*f[2*j];
      fY[j] = 100*f[2*j+1];
    }
  }

  cvSobel(flowX, flowXdX, 1, 0, 1);
  cvSobel(flowX, flowXdY, 0, 1, 1);
  cvSobel(flowY, flowYdX, 1, 0, 1);
  cvSobel(flowY, flowYdY, 0, 1, 1);
  
  BuildDescMat(flowXdX, flowXdY, descMatX, descInfo);
  BuildDescMat(flowYdX, flowYdY, descMatY, descInfo);
  
  cvReleaseImage(&flowX);
  cvReleaseImage(&flowY);
  cvReleaseImage(&flowXdX);
  cvReleaseImage(&flowXdY);
  cvReleaseImage(&flowYdX);
  cvReleaseImage(&flowYdY);
}

/* tracking interest points by median filtering in the optical field */
void OpticalFlowTracker(IplImage* flow, // the optical field
			std::vector<CvPoint2D32f>& points_in, // input interest point positions
			std::vector<CvPoint2D32f>& points_out, // output interest point positions
			std::vector<int>& status) // status for successfully tracked or not
{
  if(points_in.size() != points_out.size())
    fprintf(stderr, "the numbers of points don't match!");
  if(points_in.size() != status.size())
    fprintf(stderr, "the number of status doesn't match!");
  int width = flow->width;
  int height = flow->height;
  
  for(int i = 0; i < points_in.size(); i++) {
    CvPoint2D32f point_in = points_in[i];
    std::list<float> xs;
    std::list<float> ys;
    int x = cvFloor(point_in.x);
    int y = cvFloor(point_in.y);
    for(int m = x-1; m <= x+1; m++)
      for(int n = y-1; n <= y+1; n++) {
	int p = std::min<int>(std::max<int>(m, 0), width-1);
	int q = std::min<int>(std::max<int>(n, 0), height-1);
	const float* f = (const float*)(flow->imageData + flow->widthStep*q);
	xs.push_back(f[2*p]);
	ys.push_back(f[2*p+1]);
      }
    
    xs.sort();
    ys.sort();
    int size = xs.size()/2;
    for(int m = 0; m < size; m++) {
      xs.pop_back();
      ys.pop_back();
    }
    
    CvPoint2D32f offset;
    offset.x = xs.back();
    offset.y = ys.back();
    CvPoint2D32f point_out;
    point_out.x = point_in.x + offset.x;
    point_out.y = point_in.y + offset.y;
    points_out[i] = point_out;
    if( point_out.x > 0 && point_out.x < width && point_out.y > 0 && point_out.y < height )
      status[i] = 1;
    else
      status[i] = -1;
  }
}

/* check whether a trajectory is valid or not */
int isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length,
	    float min_var, float max_var, float max_dis){
  int size = track.size();
  for(int i = 0; i < size; i++) {
    mean_x += track[i].x;
    mean_y += track[i].y;
  }
  mean_x /= size;
  mean_y /= size;

  for(int i = 0; i < size; i++) {
    track[i].x -= mean_x;
    var_x += track[i].x*track[i].x;
    track[i].y -= mean_y;
    var_y += track[i].y*track[i].y;
  }
  var_x /= size;
  var_y /= size;
  var_x = sqrt(var_x);
  var_y = sqrt(var_y);
  // remove static trajectory
  if(var_x < min_var && var_y < min_var)
    return 0;
  // remove random trajectory
  if( var_x > max_var || var_y > max_var )
    return 0;
  
  for(int i = 1; i < size; i++) {
    float temp_x = track[i].x - track[i-1].x;
    float temp_y = track[i].y - track[i-1].y;
    length += sqrt(temp_x*temp_x+temp_y*temp_y);
    track[i-1].x = temp_x;
    track[i-1].y = temp_y;
  }
  
  float len_thre = length*0.7;
  for( int i = 0; i < size-1; i++ ) {
    float temp_x = track[i].x;
    float temp_y = track[i].y;
    float temp_dis = sqrt(temp_x*temp_x + temp_y*temp_y);
    if( temp_dis > max_dis && temp_dis > len_thre )
      return 0;
  }
  
  track.pop_back();
  // normalize the trajectory
  for(int i = 0; i < size-1; i++) {
    track[i].x /= length;
    track[i].y /= length;
  }
  return 1;
}

/* detect new feature points in the whole image */
void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points,
		   const double quality, const double min_distance){
  int width = cvFloor(grey->width/min_distance);
  int height = cvFloor(grey->height/min_distance);
  double maxVal = 0;
  cvCornerMinEigenVal(grey, eig, 3, 3);
  cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
  const double threshold = maxVal*quality;
  
  int offset = cvFloor(min_distance/2);
  for(int i = 0; i < height; i++) 
    for(int j = 0; j < width; j++) {
      int x = cvFloor(j*min_distance+offset);
      int y = cvFloor(i*min_distance+offset);
      if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
	points.push_back(cvPoint2D32f(x,y));
    }
}
/* detect new feature points in a image without overlapping to previous points */
void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,
		   std::vector<CvPoint2D32f>& points_out, const double quality, const double min_distance){
  int width = cvFloor(grey->width/min_distance);
  int height = cvFloor(grey->height/min_distance);
  double maxVal = 0;
  cvCornerMinEigenVal(grey, eig, 3, 3);
  cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
  const double threshold = maxVal*quality;

  std::vector<int> counters(width*height);
  for(int i = 0; i < points_in.size(); i++) {
    CvPoint2D32f point = points_in[i];
    if(point.x >= min_distance*width || point.y >= min_distance*height)
      continue;
    int x = cvFloor(point.x/min_distance);
    int y = cvFloor(point.y/min_distance);
    counters[y*width+x]++;
  }
  
  int index = 0;
  int offset = cvFloor(min_distance/2);
  for(int i = 0; i < height; i++) 
    for(int j = 0; j < width; j++, index++) {
      if(counters[index] == 0) {
	int x = cvFloor(j*min_distance+offset);
	int y = cvFloor(i*min_distance+offset);
	if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
	  points_out.push_back(cvPoint2D32f(x,y));
      }
    }
}
/* Initialize ! (c'était dans le fichier Initialize.h à voir comment faire pour modulariser */
void InitTrackerInfo(TrackerInfo* tracker, int track_length, int init_gap){
  tracker->trackLength = track_length;
  tracker->initGap = init_gap;
}

DescMat* InitDescMat(int height, int width, int nBins)
{
  DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
  descMat->height = height;
  descMat->width = width;
  descMat->nBins = nBins;
  descMat->desc = (float*)malloc(height*width*nBins*sizeof(float));
  memset( descMat->desc, 0, height*width*nBins*sizeof(float));
  return descMat;
}

void ReleDescMat( DescMat* descMat){
  free(descMat->desc);
  free(descMat);
}

void InitDescInfo(DescInfo* descInfo, int nBins, int flag, int orientation, int size, int nxy_cell, int nt_cell, float min_flow){
  descInfo->nBins = nBins;
  descInfo->fullOrientation = orientation;
  descInfo->norm = 2;
  descInfo->threshold = min_flow;
  descInfo->flagThre = flag;
  descInfo->nxCells = nxy_cell;
  descInfo->nyCells = nxy_cell;
  descInfo->ntCells = nt_cell;
  descInfo->dim = descInfo->nBins*descInfo->nxCells*descInfo->nyCells;
  descInfo->blockHeight = size;
  descInfo->blockWidth = size;
}

void usage(){
  fprintf(stderr, "Extract dense trajectories from a video\n\n");
  fprintf(stderr, "Usage: DenseTrack video_file [options]\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h                        Display this message and exit\n");
  fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
  fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
  fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
  fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
  fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
  fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
  fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
}

/*void arg_parse(int argc, char** argv)
  {
  int c;
  char* executable = basename(argv[0]);
  while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:")) != -1)
  switch(c) {
  case 'S':
  start_frame = atoi(optarg);
  break;
  case 'E':
  end_frame = atoi(optarg);
  break;
  case 'L':
  track_length = atoi(optarg);
  break;
  case 'W':
  min_distance = atoi(optarg);
  break;
  case 'N':
  patch_size = atoi(optarg);
  break;
  case 's':
  nxy_cell = atoi(optarg);
  break;
  case 't':
  nt_cell = atoi(optarg);
  break;

  case 'h':
  usage();
  exit(0);
  break;

  default:
  fprintf(stderr, "error parsing arguments at -%c\n  Try '%s -h' for help.", c, executable );
  abort();
  }
  }*/

/**
 * \fn int extractFeaturePoints(std::string video, int dim, int maxPts, KMdata* dataPts)
 * \brief Permits to extract STIPs from a video .avi. It save the MBH of the trajectories in the object KMdata.
 *
 * \param[in] stip Name of the video.
 * \param[in] dim STIPs dimension.
 * \param[in] maxPts Maximum number of points we want to use.
 * \param[out] dataPts The object in which we save the STIPs.
 * \return Number of points extracted.
 */
int extract_feature_points(std::string video,
			   int scale_num,
			   std::string descriptor,
			   int dim,
			   int maxPts,
			   KMdata& dataPts){
  int frameNum = 0;
  TrackerInfo tracker;
  DescInfo hogInfo;
  DescInfo hofInfo;
  DescInfo mbhInfo;
  
  IplImageWrapper image, prev_image, grey, prev_grey;
  IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;
  
  CvCapture* capture = 0;
  float* fscales = 0; // float scale values
  int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
  
  // parameters for descriptors
  int patch_size = 32; // The neighborhood size for computing the descriptor (default: N=32 pixels)
  int nxy_cell = 2; // The number of cells in the nxy axis (default: nxy=2 cells)\n");
  int nt_cell = 3; // The number of cells in the nt axis (default: nt=3 cells)\n");
  bool fullOrientation = true;
  float epsilon = 0.05;
  const float min_flow = 0.4*0.4;
  //const float PI = 3.14159;
  
  // parameters for tracking
  int start_frame = 0;
  int end_frame = 1000000;
  double quality = 0.001; // default 0.001
  double min_distance = 5; // The stride for dense sampling feature points (default: W=5 pixels)
  int init_gap = 1;
  int track_length = 15; //  The length of the trajectory (default: L=15 frames)
  
  // parameters for the trajectory descriptor
  const float min_var = sqrt(3);
  const float max_var = 50;
  const float max_dis = 20;
   
  // parameters for multi-scale
  //int scale_num = 0; (in [1,8])
  const float scale_stride = sqrt(2);
  
  //arg_parse(argc, argv);
  
  //std::cerr << "start_frame: " << start_frame << " end_frame: " << end_frame << " track_length: " << track_length << std::endl;
  //std::cerr << "min_distance: " << min_distance << " patch_size: " << patch_size << " nxy_cell: " << nxy_cell << " nt_cell: " << nt_cell << std::endl;
  
  InitTrackerInfo(&tracker, track_length, init_gap);
  InitDescInfo(&hogInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell, min_flow);
  InitDescInfo(&hofInfo, 9, 1, 1, patch_size, nxy_cell, nt_cell, min_flow);
  InitDescInfo(&mbhInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell, min_flow);
  
  capture = cvCreateFileCapture(video.c_str());
  
  if( !capture ) { 
    printf( "Could not initialize capturing..\n" );
    exit(EXIT_FAILURE);
  }
  

  if( show_track == 1 )
    cvNamedWindow( "DenseTrack", 0 );
  
  std::vector<std::list<Track> > xyScaleTracks;
  int init_counter = 0; // indicate when to detect new feature points
  int nPts = 0; // actual number of points
  while( true ) {
    IplImage* frame = 0;
    int i, j, c;
    
    // get a new frame
    frame = cvQueryFrame( capture );
    if( !frame ) {
      //printf("break");
      break;
    }
    if( frameNum >= start_frame && frameNum <= end_frame ) {
      if( !image ) {
	// initailize all the buffers
	image = IplImageWrapper( cvGetSize(frame), 8, 3 );
	image->origin = frame->origin;
	prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
	prev_image->origin = frame->origin;
	grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
	grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
	prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
	prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
	eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, scale_stride );
		
	cvCopy( frame, image, 0 );
	cvCvtColor( image, grey, CV_BGR2GRAY );
	grey_pyramid.rebuild( grey );
		
	// how many scale we can have
	scale_num = std::min<std::size_t>(scale_num, grey_pyramid.numOfLevels());
	fscales = (float*)cvAlloc(scale_num*sizeof(float));
	xyScaleTracks.resize(scale_num);
		
	for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {
	  std::list<Track>& tracks = xyScaleTracks[ixyScale];
	  fscales[ixyScale] = pow(scale_stride, ixyScale);
		
	  // find good features at each scale separately
	  IplImage *grey_temp = 0, *eig_temp = 0;
	  std::size_t temp_level = (std::size_t)ixyScale;
	  grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
	  eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
	  std::vector<CvPoint2D32f> points(0);
	  cvDenseSample(grey_temp, eig_temp, points, quality, min_distance);
		  
	  // save the feature points
	  for( i = 0; i < points.size(); i++ ) {
	    Track track(tracker.trackLength);
	    PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
	    track.addPointDesc(point);
	    tracks.push_back(track);
	  }
		  
	  cvReleaseImage( &grey_temp );
	  cvReleaseImage( &eig_temp );
	}
      }
      
      // build the image pyramid for the current frame
      cvCopy( frame, image, 0 );
      cvCvtColor( image, grey, CV_BGR2GRAY );
      grey_pyramid.rebuild(grey);
      
      if( frameNum > 0 ) {
	init_counter++;
	for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {
	  // track feature points in each scale separately
	  std::vector<CvPoint2D32f> points_in(0);
	  std::list<Track>& tracks = xyScaleTracks[ixyScale];
	  for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack) {
	    CvPoint2D32f point = iTrack->pointDescs.back().point;
	    points_in.push_back(point); // collect all the feature points
	  }
	  int count = points_in.size();
	  IplImage *prev_grey_temp = 0, *grey_temp = 0;
	  std::size_t temp_level = ixyScale;
	  prev_grey_temp = cvCloneImage(prev_grey_pyramid.getImage(temp_level));
	  grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
		  
	  cv::Mat prev_grey_mat = cv::cvarrToMat(prev_grey_temp);
	  cv::Mat grey_mat = cv::cvarrToMat(grey_temp);
		  
	  std::vector<int> status(count);
	  std::vector<CvPoint2D32f> points_out(count);
		  
	  // compute the optical flow
	  IplImage* flow = cvCreateImage(cvGetSize(grey_temp), IPL_DEPTH_32F, 2);
	  cv::Mat flow_mat = cv::cvarrToMat(flow);
	  cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,
					sqrt(2)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
	  // track feature points by median filtering
	  OpticalFlowTracker(flow, points_in, points_out, status);
		  
	  int width = grey_temp->width;
	  int height = grey_temp->height;
	  
	  // Computing histograms
	  DescMat* hogMat = NULL;
	  DescMat* hofMat = NULL;
	  DescMat* mbhMatX = NULL;
	  DescMat* mbhMatY = NULL;
	  if(descriptor.compare("hoghof") == 0 || descriptor.compare("all") == 0){
	    hogMat = InitDescMat(height, width, hogInfo.nBins);
	    HogComp(prev_grey_temp, hogMat, hogInfo);
	    
	    hofMat = InitDescMat(height, width, hofInfo.nBins);
	    HofComp(flow, hofMat, hofInfo);
	  }
	  if(descriptor.compare("mbh") == 0 || descriptor.compare("all") == 0){
	    mbhMatX = InitDescMat(height, width, mbhInfo.nBins);
	    mbhMatY = InitDescMat(height, width, mbhInfo.nBins);
	    MbhComp(flow, mbhMatX, mbhMatY, mbhInfo);
	  }
	  
	  i = 0;
	  for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i) {
	    if( status[i] == 1 ) { // if the feature point is successfully tracked
	      PointDesc& pointDesc = iTrack->pointDescs.back();
	      CvPoint2D32f prev_point = points_in[i];
	      // get the descriptors for the feature point
	      CvScalar rect;
	      if(descriptor.compare("hoghof") == 0 || descriptor.compare("all") == 0){
		rect = getRect(prev_point, cvSize(width, height), hogInfo);
		pointDesc.hog = getDesc(hogMat, rect, hogInfo, epsilon);
		pointDesc.hof = getDesc(hofMat, rect, hofInfo, epsilon);
	      }
	      if(descriptor.compare("mbh") == 0 || descriptor.compare("all") == 0){
		rect = getRect(prev_point, cvSize(width, height), mbhInfo);
		pointDesc.mbhX = getDesc(mbhMatX, rect, mbhInfo, epsilon);
		pointDesc.mbhY = getDesc(mbhMatY, rect, mbhInfo, epsilon);
	      }
	      PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
	      iTrack->addPointDesc(point);
		      
	      // draw this track
	      if( show_track == 1 ) {
		std::list<PointDesc>& descs = iTrack->pointDescs;
		std::list<PointDesc>::iterator iDesc = descs.begin();
		float length = descs.size();
		CvPoint2D32f point0 = iDesc->point;
		point0.x *= fscales[ixyScale]; // map the point to first scale
		point0.y *= fscales[ixyScale];
			
		float j = 0; 
		for (iDesc++; iDesc != descs.end(); ++iDesc, ++j) {
		  CvPoint2D32f point1 = iDesc->point;
		  point1.x *= fscales[ixyScale];
		  point1.y *= fscales[ixyScale];
		
		  cvLine(image, cvPointFrom32f(point0), cvPointFrom32f(point1),
			 CV_RGB(0,cvFloor(255.0*(j+1.0)/length),0), 2, 8,0);
		  point0 = point1;
		}
		cvCircle(image, cvPointFrom32f(point0), 2, CV_RGB(255,0,0), -1, 8,0);
	      }
	      ++iTrack;
	    }
	    else // remove the track, if we lose feature point
	      iTrack = tracks.erase(iTrack);
	  }
	  // Releasing memory
	  if(descriptor.compare("hoghof") == 0 || descriptor.compare("all") == 0){
	    ReleDescMat(hogMat);
	    ReleDescMat(hofMat);
	  }
	  if(descriptor.compare("mbh") == 0 || descriptor.compare("all") == 0){
	    ReleDescMat(mbhMatX);
	    ReleDescMat(mbhMatY);
	  }
	  cvReleaseImage( &prev_grey_temp );
	  cvReleaseImage( &grey_temp );
	  cvReleaseImage( &flow );
	}

	for( int ixyScale = 0; ixyScale < scale_num; ++ixyScale ) {
	  std::list<Track>& tracks = xyScaleTracks[ixyScale]; // output the features for each scale
	  for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ) {
	    if( iTrack->pointDescs.size() >= tracker.trackLength+1 ) { // if the trajectory achieves the length we want
	      std::vector<CvPoint2D32f> trajectory(tracker.trackLength+1);
	      std::list<PointDesc>& descs = iTrack->pointDescs;
	      std::list<PointDesc>::iterator iDesc = descs.begin();
	      
	      for (int count = 0; count <= tracker.trackLength; ++iDesc, ++count) {
		trajectory[count].x = iDesc->point.x*fscales[ixyScale];
		trajectory[count].y = iDesc->point.y*fscales[ixyScale];
	      }
	      float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
	      if( isValid(trajectory, mean_x, mean_y, var_x, var_y, length, min_var, max_var, max_dis) == 1 ) {
		//printf("%d\t", frameNum);
		//printf("%f\t%f\t", mean_x, mean_y);
		//printf("%f\t%f\t", var_x, var_y);
		//printf("%f\t", length);
		//printf("%f\t", fscales[ixyScale]);
				
		//for (int count = 0; count < tracker.trackLength; ++count)
		//printf("%f\t%f\t", trajectory[count].x,trajectory[count].y );
				
		int d = 0; // to fill dataPts 
		
		// COMPUTE HOG HOG
		if(descriptor.compare("hoghof") == 0 || descriptor.compare("all") == 0){
		  iDesc = descs.begin();
		  int t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);
		  for( int n = 0; n < hogInfo.ntCells; n++ ) {
		    std::vector<float> vec(hogInfo.dim);
		    for( int t = 0; t < t_stride; t++, iDesc++ )
		    for( int m = 0; m < hogInfo.dim; m++ )
		      vec[m] += iDesc->hog[m];
		    for( int m = 0; m < hogInfo.dim; m++ ){
		      // printf("%f\t", vec[m]/float(t_stride));
		      dataPts[nPts][d] = vec[m]/float(t_stride);
		      d++;
		    }
		  }
		  
		  iDesc = descs.begin();
		  t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
		  for( int n = 0; n < hofInfo.ntCells; n++ ) {
		    std::vector<float> vec(hofInfo.dim);
		    for( int t = 0; t < t_stride; t++, iDesc++ )
		      for( int m = 0; m < hofInfo.dim; m++ )
			vec[m] += iDesc->hof[m];
		    for( int m = 0; m < hofInfo.dim; m++ ){
		      //printf("%f\t", vec[m]/float(t_stride));
		      dataPts[nPts][d] = vec[m]/float(t_stride);
		      d++;
		    }
		  }
		}
		
		// COMPUTE MBHX AND MBHY
		if(descriptor.compare("mbh") == 0 || descriptor.compare("all") == 0){
		  iDesc = descs.begin();
		  int t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
		  for( int n = 0; n < mbhInfo.ntCells; n++ ) {
		    std::vector<float> vec(mbhInfo.dim);
		    for( int t = 0; t < t_stride; t++, iDesc++ )
		      for( int m = 0; m < mbhInfo.dim; m++ )
			vec[m] += iDesc->mbhX[m];
		    for( int m = 0; m < mbhInfo.dim; m++ ){
		      dataPts[nPts][d] = vec[m]/float(t_stride);
		      d++;
		    }
		  }
		  
		  iDesc = descs.begin();
		  t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
		  for( int n = 0; n < mbhInfo.ntCells; n++ ) {
		    std::vector<float> vec(mbhInfo.dim);
		    for( int t = 0; t < t_stride; t++, iDesc++ )
		      for( int m = 0; m < mbhInfo.dim; m++ )
			vec[m] += iDesc->mbhY[m];
		    
		    for( int m = 0; m < mbhInfo.dim; m++ ){
		      dataPts[nPts][d] = vec[m]/float(t_stride);
		      d++;
		    }
		  }
		}
		
		// Following vector
		nPts++;
	      }
	      iTrack = tracks.erase(iTrack);
	    }
	    else
	      iTrack++;
	  }
	}
	
	if( init_counter == tracker.initGap ) { // detect new feature points every initGap frames
	  init_counter = 0;
	  for (int ixyScale = 0; ixyScale < scale_num; ++ixyScale) {
	    std::list<Track>& tracks = xyScaleTracks[ixyScale];
	    std::vector<CvPoint2D32f> points_in(0);
	    std::vector<CvPoint2D32f> points_out(0);
	    for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++) {
	      std::list<PointDesc>& descs = iTrack->pointDescs;
	      CvPoint2D32f point = descs.back().point; // the last point in the track
	      points_in.push_back(point);
	    }
	    
	    IplImage *grey_temp = 0, *eig_temp = 0;
	    std::size_t temp_level = (std::size_t)ixyScale;
	    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
	    eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
	    
	    cvDenseSample(grey_temp, eig_temp, points_in, points_out, quality, min_distance);
	    // save the new feature points
	    for( i = 0; i < points_out.size(); i++) {
	      Track track(tracker.trackLength);
	      PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
	      track.addPointDesc(point);
	      tracks.push_back(track);
	    }
	    cvReleaseImage( &grey_temp );
	    cvReleaseImage( &eig_temp );
	  }
	}
      }
      
      cvCopy( frame, prev_image, 0 );
      cvCvtColor( prev_image, prev_grey, CV_BGR2GRAY );
      prev_grey_pyramid.rebuild(prev_grey);
    }
    
    if( show_track == 1 ) {
      cvShowImage( "DenseTrack", image);
      c = cvWaitKey(3);
      if((char)c == 27) break;
    }
    // get the next frame
    frameNum++;
  }
  
  if( show_track == 1 )
    cvDestroyWindow("DenseTrack");
  return nPts;
}
