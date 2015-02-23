#ifndef _IMCONFIG_H
#define _IMCONFIG_H

#include <stdlib.h> 
#include <dirent.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

/** \class activitiesMap
 * \brief Correspondance between a label and an activity
 */
class activitiesMap{
 public:
  int label;
  std::string activity;
};

void listBdds();
void listActivities(std::string bdd);
int mapActivities(std::string path2bdd, activitiesMap** am);
bool labelExist(int label, activitiesMap *am, int nbActivities);
int searchMapIndex(int label, activitiesMap *am, int nbActivities);

// Dimension of the feature points
int getDim(std::string descriptor);
void saveDescInfo(std::string folder, int desc);
int getDescID(std::string folder);

// Probabiliy Estimation
double getTrainProbability(std::string folder);
void saveTrainProbability(std::string folder, double p);

// Number of centers
int getK(std::string kmeans);
void saveKinfo(std::string folder, int k);

#endif // _IMCONFIG_H
