/** @author Fabien ROUALDES (institut Mines-Télécom)
 *  @author HE Huilong (institut Mines-Télécom)
 *  @file  main.c
 *  @date 20/09/2013
 *  Program permiting to use kmeans and svm without writing on hard drive and linked to dynamic libraries
 */
#include "integration.h"

using namespace std;

std::string integration(std::string video, std::string folder){

  std::string path2bdd(folder);   
  std::string bddName("activity_recognition"); 
  // Loading parameters
  IMbdd bdd(bddName,path2bdd);
  bdd.load_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
  int scale_num = bdd.getScaleNum();
  std::string descriptor = bdd.getDescriptor();
  int dim = bdd.getDim();
  int k = bdd.getK();
  std::cout << "dim: "<< dim << " k: " << k << " scale: " << scale_num<<std::endl;
  //int maxPts = bdd.getMaxPts();
  int maxPts = 10000; 
  // Computing feature points
  KMdata dataPts(dim,maxPts);
  int nPts = 0;
  nPts = extract_feature_points(video,
				scale_num, descriptor, dim,
				maxPts, dataPts);		
  if(nPts == 0){
    std::cerr << "No activity detected !" << std::endl;
    return "Nothing happened";
  }
  std::cout << nPts << " feature vectors extracted..." << std::endl;
  dataPts.setNPts(nPts);
  dataPts.buildKcTree();
  
  KMfilterCenters ctrs(k, dataPts);  
  std::string pathToCenters = bdd.getFolder() + "/" + bdd.getKMeansFile();
  std::cout << pathToCenters << std::endl;
  importCenters(bdd.getFolder() + "/" + bdd.getKMeansFile(), dim, k, &ctrs);
  std::cout << "KMeans centers imported..." << std::endl;
  
  /*
  activitiesMap *am;
  int nbActivities = mapActivities(path2bdd,&am);
  */ 

  struct svm_problem svmProblem = computeBOW(0,
					     dataPts,
					     ctrs);

  // simple, gaussian, both, nothing
  if(bdd.getNormalization().compare("") != 0){
    bow_normalization(bdd,svmProblem);
    std::cout << "Bag of words normalized..." << std::endl;
  }

  std::vector<std::string> activities = bdd.getActivities(); 
  int nbActivities = activities.size();
  struct svm_model** pSVMModels = new svm_model*[nbActivities];
  std::vector<std::string> modelFiles(bdd.getModelFiles());
  int i=0;
  for (std::vector<std::string>::iterator it = modelFiles.begin() ; it != modelFiles.end() ; ++it){
    std::cout << (*it) << std::endl;
    pSVMModels[i]= svm_load_model((*it).c_str());
    i++;
  }
  std::cout << "SVM models imported..." << std::endl;
  double probs[nbActivities];
  double label = svm_predict_ovr_probs(pSVMModels,
				       svmProblem.x[0],
				       nbActivities,
				       probs,
				       2);
  std::cerr<<"Probs: ";
  for(int j=0 ; j<nbActivities ; j++){
    std::cout << setw(2) << setiosflags(ios::fixed) << probs[j]<<" "; 
  }
  std::cout << std::endl;

  //int index = searchMapIndex(label, am, nbActivities);
  std::cout << "Activity predicted: ";
  int label_int = 1;
  while(label_int - label != 0) label_int++ ;
  std::cout << label_int << "(" << activities[label_int-1] << ")";
  //std::cout << am[index].activity << "(" << am[index].label << ")";
  std::cout << std::endl;
  
  for(int m=0 ; m<nbActivities ; m++){
    svm_free_and_destroy_model(&pSVMModels[m]);
  }
  delete[] pSVMModels;
  destroy_svm_problem(svmProblem);

  return activities[label_int-1];
}
void printTime(exec_time *tmps){
  int top = sysconf(_SC_CLK_TCK); // number of tips per seconds
  std::cout << "[" << (tmps->end - tmps->begin)/top << "s]" <<std::endl;
}
