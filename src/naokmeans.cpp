/** 
 * \file naokmeans.cpp
 * \brief  Set of functions permiting to execute KMeans algorithms
 * using KMlocal classes.
 * \author Fabien ROUALDES (institut Mines-Télécom)
 * \date 02/07/2013 
 *
 */
#include "naokmeans.h"
#include <time.h>

/**
 * \fn int importSTIPs(std::string stip, int dim, int maxPts, KMdata* dataPts)
 * \brief STIPs importation function in the format 1 point = 1 line.
 * Each dimension are separated from one space (" ").
 *
 * \param[in] stip Name of the file containing the STIPs.
 * \param[in] dim The STIPs dimension.
 * \param[in] maxPts The maximum number of points you want to import.
 * \param[out] dataPts The KMlocal object which will be containing STIPs.
 * \return Number of points imported.
 */
int importSTIPs(std::string stip, int dim, int maxPts, KMdata* dataPts){
  int nPts = 0; // actual number of points

  ifstream in(stip.c_str(), ios::in);	
  if (!in){
    cerr << "Pas de données à lire !!!" << endl;
    exit(EXIT_FAILURE);
  }  
  bool endOfLine = false;
  // Saving each vectors
  while(!in.eof() && nPts < maxPts){
    // Saving each dimension
    int d = 0;
    while(!endOfLine && d < dim){
      if(!(in >> (*dataPts)[nPts][d])){ // we save the stream in the buffer and 
	endOfLine = true; // if there is no more character it is because we are at the end of the line.
      }
      d++;
    }
    nPts++;
  }
  return nPts-1;
}

/**
 * \fn void exportSTIPs(std::string stip, int dim, const KMdata& dataPts)
 * \brief STIPs exportation function in the format 1 point = 1 line.
 * Each dimension are separated from one space (" ").
 *
 * \param[in] stip Name of the file containing the STIPs.
 * \param[in] dim The STIPs dimension.
 * \param[in] dataPts The KMlocal object which will be containing STIPs.
 */
void exportSTIPs(std::string stip, int dim, const KMdata& dataPts){
  int nPts = dataPts.getNPts(); // actual number of points
  
  // open fiouverture en écriture avec effacement du fichier ouvert
  ofstream sSTIPs(stip.c_str(), ios::out | ios::trunc);
  if(!sSTIPs){
    std::cerr << "Impossible d'ouvrir le fichier !" << std::endl;
    exit(EXIT_FAILURE);
  }
  for(int i=0; i<nPts ;i++){
    for(int d = 0; d<dim ; d++){
      sSTIPs << dataPts[i][d] << " ";
    }
    sSTIPs << std::endl;
  }
  sSTIPs.close();
}

/**
 * \fn void exportCenters(std::string centers, int dim, int k, KMfilterCenters ctrs)
 * \brief Export function to save KMfilterCenters in a file.
 * One line corresponds to one point with dim value (separeted from one space " ").
 *
 * \param[in] centers Name of the file which will be containing dimensions of each centers.
 * \param[in] dim Center's dimension.
 * \param[in] k Number of centers.
 * \param[in] ctrs The centers.
 */
void exportCenters(std::string centers, int dim, int k, KMfilterCenters ctrs){
  // ouverture en écriture avec effacement du fichier ouvert
  ofstream trainingMeans(centers.c_str(), ios::out | ios::trunc);
  if(!trainingMeans){
    cerr << "Impossible d'ouvrir le fichier !" << endl;
    exit(EXIT_FAILURE);
  }
  for(int i=0; i<k ;i++){
    KMpoint aux = ctrs[i];
    for(int d = 0; d<dim ; d++){
      trainingMeans << aux[d] << " ";
    }
    trainingMeans << endl;
  }
  trainingMeans.close();
}

/**
 * \fn void importCenters(std::string centers, int dim, int k, KMfilterCenters* ctrs)
 * \brief Importation function saving external centers in the KMfilterCenters object.
 * One line corresponds to one centers with its values (separeted from one space " ").
 *
 * \param[in] centers Name of the file which will be containing dimensions of each centers.
 * \param[in] dim Center's dimension.
 * \param[in] k Number of centers.
 * \param[out] ctrs The centers.
 */
void importCenters(std::string centers, int dim, int k, KMfilterCenters* ctrs){
  ifstream in(centers.c_str(), ios::in);	
  if (!in){
    cerr << "Pas de données à lire !!!" << endl;
    exit(EXIT_FAILURE);
  }
  bool endOfLine = false;
  int center = 0;
  while(!(in.eof()) && center < k){
    // saving each means 
    int d = 0;
    while(!endOfLine && d < dim){
      if(!(in >> (*ctrs)[center][d])){
	endOfLine = true;
      }
      d++;
    }
    center++;
  }
}

/**
 * \fn void kmIvanAlgorithm(int ic, int dim,  const KMdata& dataPts, int k, KMfilterCenters& ctrs)
 * \brief This is an optimized KMeans algorithm. Ivan's algorithm uses
 * basic KMeans algorithm (here the Lloyd's one) and the idea was to 
 * initialize centers intelligently.
 *
 * \param[in] ic The iteration coefficient will determine the number of iterations in each phases.
 * \param[in] dim Points and centers's dimension.
 * \param[in] dataPts The data we want to compute the centers.
 * \param[in] k The number of centers.
 * \param[out] ctrs The centers.
 *
 * The Ivan's algorithm is divided into 3 phases. The first phase is executed on
 * 25 per cent of the data (randomly sampled). To begin, the centers are randomly generated.
 * Then ic * 4 iterations of a KMeans algorithm are executed.
 * During the second part we cluster 50 per cent of the data using the older centroids. 
 * This step is computed ic * 2 times.
 * Finally, we make ic * 1 iteration on all the data.
 *
 */
void kmIvanAlgorithm(int ic, int dim,  const KMdata& dataPts, int k, KMfilterCenters& ctrs){
  int nPts = dataPts.getNPts();
  KMdata subDataPts(dim,nPts);

  int* randomVector = NULL;
  
  double** centersBuffer = NULL;
  centersBuffer = (double **) malloc(k*sizeof(double*));
  for(int c=0 ; c<k ; c++){
    centersBuffer[c] = (double*) malloc(dim*sizeof(double));
  }
  
  // ic : iteration coefficient
  int nrPhases = 3;
  for(int i=0 ; i<nrPhases ; i++){
    std::cerr<<"phase:"<<i<<std::endl;
    int maxIter = (int) pow(2,(nrPhases-1-i));
    int sampleSize = floor(nPts/maxIter);
    
    std::cout << "Applying k-means: " << endl;
    std::cout << "Clustering " << sampleSize << " vectors (ie. " << 100/maxIter << " percent of the data)";
    std::cout << " in " << k << " clusters";
    if (i>0){
      std::cout << " using older centroids..." << std::endl;
    }
    else{
      std::cout << "..." << endl;
      std::cout << "Initializing centroids by sampling..." << std::endl;
    }

    if (i == nrPhases-1){
      // Filling subDataPts
      for(int s=0; s<nPts ; s++){
	      for(int d=0 ; d<dim ; d++){
	        subDataPts[s][d] = dataPts[s][d];
	      }
      }
      subDataPts.setNPts(nPts);
    }
    else{
      // Filling the random vector permiting to sampling "uniformly" (as more as we can) the data  
      //std::cerr<<"ok++"<<std::endl; 
      randomVector = (int*) malloc(sampleSize * sizeof(int));
      srand(time(NULL)); // initialisation of rand
      for(int s=0; s<sampleSize ;s++){
	      int r = (int) rand()%(nPts);
	      int index = 0;
	      while(index<s && randomVector[index] != r){
	        index++;
	      }
	      if(s==0 || randomVector[index] != r)
	        randomVector[s] = r;
	      else{
	        s--;
	      }
      }
      //std::cerr<<"ok--"<<std::endl; 
      // Filling subDataPts
      for(int s=0; s<sampleSize ; s++){
	      for(int d=0 ; d<dim ; d++){
	        subDataPts[s][d] = dataPts[randomVector[s]][d];
	      }
      }
      subDataPts.setNPts(sampleSize);
    }
    subDataPts.buildKcTree();
    
    // Allocate centers with subData
    KMfilterCenters newCtrs(k, subDataPts);
    
    // Initializing the centers (randomly for the first iteration)
    if(i==0){
      (newCtrs).genRandom(); 
    }
    else{
      for(int c = 0; c < k ; c++){
	      for(int d=0 ; d<dim ; d++){
	        (newCtrs)[c][d] = centersBuffer[c][d];
	      }
      } 
    }
    for(int iteration = 0  ; iteration < ic*maxIter ; iteration++){ // ic : iteration coefficient
      (newCtrs).lloyd1Stage();
    }
    
    // Saving the old centers in centersBuffer
    for(int c = 0; c < k ; c++){
      for(int d=0 ; d<dim ; d++){
	      centersBuffer[c][d] = (newCtrs)[c][d];
      }
    }
    
    if(i==nrPhases-1){
      for(int c = 0; c < k ; c++){
	      for(int d=0 ; d<dim ; d++){
	        ctrs[c][d] = centersBuffer[c][d];
	      }
      }
    }
    free(randomVector);
    randomVector = NULL;
  }

  for(int c=0 ; c<k ; c++){
    free(centersBuffer[c]);
  }
  free(centersBuffer);
}

/**
 * \fn void createTrainingMeans(std::string stipFile, int dim, int maxPts, int k, std::string meansFile)
 * \brief Import HOG and HOF from a file and compute KMeans algorithm to create
 * the file training.means.
 * 
 * \param[in] stipFile The file containing the STIPs.
 * \param[in] dim Points and centers's dimension.
 * \param[in] maxPts The maximum number of data we can compute
 * \param[in] k The number of centers
 * \param[out] meansFile The file in wich we will save the KMeans centers.
 */
void createTrainingMeans(std::string stipFile,
			 int dim,
			 int maxPts,
			 int k,
                         std::string meansFile
                         ){

  KMdata dataPts(dim,maxPts);
  int nPts = importSTIPs(stipFile, dim, maxPts, &dataPts);
  dataPts.setNPts(nPts);
  dataPts.buildKcTree();
  
  KMfilterCenters ctrs(k, dataPts);    
  
  int ic = 3;
  kmIvanAlgorithm(ic, dim, dataPts, k, ctrs);  
  
  exportCenters(meansFile, dim, k, ctrs);
}

