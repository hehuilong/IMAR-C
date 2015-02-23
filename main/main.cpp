/**
 * \function main.cpp
 * \author HUILONG He (Télécom SudParis)
 * \author ROUALDES Fabien (Télécom SudParis)
 * \date 20/09/2013
 * \brief  
 */
#include <stdlib.h>
#include <string>
#include "naomngt.h"
#include <fstream>

using namespace std;

void help();

int main(int argc, char* argv[]){
  int k;
  if(argc<3){
    help();
    return EXIT_FAILURE;
  }
  std::string function = argv[1];
  if(function.compare("refresh") == 0){ 
    // Delete all the files but videos, and recompute feature points 
    if(argc == 5){
      int scale_num = atoi(argv[3]);
      std::string desc = argv[4];
      im_refresh_bdd(argv[2],scale_num,desc);
    }
    else{
      std::cerr << "refresh: Bad arguments!" << std::endl;
      return EXIT_FAILURE;
    }
  }
  else if(function.compare("kmeans") == 0){ 
    // Do kmeans on the People set (configure the set by modifying the imconfigure.xml file)
    if(argc != 5){
      std::cerr << "kmeans: Bad arguments!" << std::endl;
      return EXIT_FAILURE;
    }
    char* bddName = argv[2];
    k = atoi(argv[3]);
    char* kmeansType = argv[4];
    im_kmeans_bdd(bddName,k,kmeansType);
  }
  else if(function.compare("learn") == 0){ // in order to add video + stips in db
    // Do cross validation and generate the svm model(one vs rest) on People set
    // Do test on the TestPeople set if it's not empty (configure the set by modifying the imconfigure.xml file)
    if(argc != 5){
      std::cerr << "learn: Bad arguments!" << std::endl;
      return EXIT_FAILURE;
    }
    char* bddName = argv[2];
    char* kernel = argv[3];
    char* norm = argv[4];
    im_train_bdd(bddName, kernel, norm);
  }
}
void help(){
  std::cout <<"===         NAO VISION TEACHER         ===" << std::endl;
  std::cout <<"=== Please read README.txt for details ===" << std::endl;
  std::cout << std::endl;
  std::cout << "Refresh the feature points by DenseTrack:" << std::endl;
  std::cout << "- Delete all the data but videos and extract feature points" << std::endl;
  std::cout << "\t ./naomngt refresh <name of database> <nombre of scales> <descriptor type> " << std::endl;
  std::cout << std::endl;
  std::cout << "KMeans:" << std::endl;
  std::cout << "\t ./naomngt kmeans <name of database> <nomber of centers> <kmeans type> " << std::endl;
  std::cout << std::endl;
  std::cout << "SVM learning:" << std::endl;
  std::cout << "\t ./naomngt learn <name of database> <kernel type> <normalization type>" << std::endl;
}
