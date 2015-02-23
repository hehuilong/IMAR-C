/**
 * \file naomngt.h
 * \author Fabien ROUALDES
 * \author Huilong HE
 * \date 20/08/2013
 */
#ifndef _NAOMNGT_H_
#define _NAOMNGT_H_ 

#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h> // mkdir 
#include <sys/types.h> // mkdir
#include <iostream>
#include <fstream> // ifstream
#include <sstream>
#include <vector>
#include <string>
#include <utility> // make_pair
#include <map>
#include <algorithm> // tri dans l'odre croissant

#include "naokmeans.h"
#include "naosvm.h"
#include "naodensetrack.h"
#include "imconfig.h"
#include "imbdd.h"

std::string inttostring(int int2str);

void im_refresh_bdd(std::string bddName,
		    int scale_num,
		    std::string descriptor);

void im_refresh_folder(const IMbdd& bdd, std::string folder, int* num_points, int* num_videos);

void im_concatenate_bdd_feature_points(std::string path2bdd,
				       std::vector<std::string> people,
				       std::vector<std::string> activities);

void im_concatenate_folder_feature_points(std::string folder,
					  std::vector<std::string> activities);

int im_create_specifics_training_means(IMbdd bdd,
				       const std::vector<std::string>& trainingPeople 
				       );

int im_create_normal_training_means(IMbdd bdd,
				       const std::vector<std::string>& trainingPeople 
				       );

double im_training_leave_one_out(const IMbdd& bdd,
				 const std::vector<std::string>& trainingPeople,
				 const std::map <std::string, struct svm_problem>& peopleBOW,
				 int& minC, int& maxC,
				 int& minG, int& maxG,
				 struct svm_parameter& svmParameter);

double im_svm_train(IMbdd& bdd,
		    const std::vector<std::string>& trainingPeople,
		    MatrixC& trainMC,
		    const std::vector<std::string>& testingPeople,
		    MatrixC& testMC,
        std::string kernel, std::string normalization);

void im_kmeans_bdd(std::string bddName, int k, std::string kmeansType);

void im_train_bdd(std::string bddName, std::string kernel, std::string normalization);

void im_compute_bdd_bow(const IMbdd& bdd, 
			std::map <std::string, struct svm_problem>& peopleBOW);

void im_normalize_bdd_bow(const IMbdd& bdd,
			  const std::vector<std::string>& people,
			  std::map<std::string, struct svm_problem>& peopleBOW);

void im_compute_gaussian_param(const IMbdd& bdd, const std::vector<std::string>& trainingPeople,
			  std::map<std::string, struct svm_problem>& peopleBOW);

void bow_normalization(const IMbdd& bdd, struct svm_problem& svmProblem);

void im_fill_confusion_matrix(const IMbdd& bdd,
			      const svm_problem& svmProblem,
			      struct svm_model** svmModels,
			      MatrixC& MC);

void compute_rbfchis_A(const std::vector<std::string> &trainingPeople, 
                         struct svm_parameter &svmParameter,
                         const std::map <std::string, struct svm_problem>& peopleBOW);
 
double chi_square_dist(const svm_node *px, const svm_node *py);

#endif
