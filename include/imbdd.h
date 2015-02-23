#ifndef _IMBDD_H_
#define _IMBDD_H_

#include <iostream>
#include <string>
#include <vector>
#include "tinyxml.h"

class IMbdd{
 private:
  std::string bddName;
  std::string folder;
  
  // Data
  int nrActivities;
  std::vector<std::string> activities;
  std::vector<std::string> people;
  std::vector<std::string> testPeople;
  std::string reject;
  
  // DenseTrack
  int scale_num;
  std::string descriptor;
  int dim;
  
  // KMeans
  int maxPts;
  std::string km_algorithm;
  int k;
  std::string KMeansFile;
  
  // Normalization
  std::string normalization;
  std::string meansFile;
  std::string standardDeviationFile;
  
  // SVM
  int nr_class;
  std::vector<std::string> modelFiles;
  
 public:
  IMbdd(std::string bddName, std::string folder);
  ~IMbdd(){};
  void write_bdd_configuration(std::string pFolder, std::string pFilename);
  void load_bdd_configuration(std::string pFolder, std::string pFilename);
  void show_bdd_configuration();
  
  std::string getName() const {return bddName;};
  std::string getFolder() const {return folder;};
  std::vector<std::string> getPeople() const {return people;};
  std::vector<std::string> getTestPeople() const {return testPeople;};
  std::vector<std::string> getActivities() const {return activities;};
  std::string getReject() const {return reject;};
  
  void saveName(std::string bddName);
  int getScaleNum() const {return scale_num;};
  std::string getDescriptor() const {return descriptor;};
  std::string getKMAlgorithm() const {return km_algorithm;};
  int getK() const {return k;}
  int getDim() const {return dim;};
  int getMaxPts() const {return maxPts;};
  std::string getKMeansFile() const {return KMeansFile;};
  std::string getNormalization() const {return normalization;};
  std::string getMeansFile() const {return meansFile;};
  std::string getStandardDeviationFile() const {return standardDeviationFile;};
  int getNrClass() const {return nr_class;};
  std::vector<std::string> getModelFiles() const {return modelFiles;};
  void changeDataSettings(std::vector<std::string> activities,
				 std::vector<std::string> people,
				 std::vector<std::string> testPeople,
				 std::string reject);
  void changeDenseTrackSettings(int scale_num,
				std::string descriptor,
				int dim);
  void changeKMSettings(std::string algorithm,
			int k,
			std::string KMeansFile);
  void changeNormalizationSettings(std::string normalization,
				   std::string meansFile,
				   std::string standardDeviationFile);  
  void changeSVMSettings(int nr_class,
			 std::vector<std::string> modelFiles);
};

#endif // _IMBDD_H_
