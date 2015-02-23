/** \author Fabien ROUALDES (institut Mines-Télécom)
 *  \file naosvm.h
 *  \date 09/07/2013 
 *  Set of function permiting to import/ predict a svm problem, import/create a svm model
 */
#ifndef _NAOSVM_H_
#define _NAOSVM_H_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "svm.h"
#include "KMlocal.h"
#include "imbdd.h"

using namespace std;

struct svm_problem computeBOW(int label, const KMdata& dataPts, KMfilterCenters& ctrs);

void destroy_svm_problem(struct svm_problem svmProblem);

void addBOW(struct svm_node* bow, double label,
	    struct svm_problem& svmProblem);

//confusion matrix
class MatrixC{
 public:
  MatrixC(const svm_model* model);
  MatrixC(int nr_class, int* labels);
  ~MatrixC();
  void output();
  void exportMC(std::string folder, std::string file);
  void calculFrequence();
  int getIndex(double lab);
  void addTransfer(double lab_in,double lab_out);
  double** getMatrix();
 private:
  int** m;
  double* labels;
  double** m_fre;
 public:
  int num_labels;
  int nrTest;
  int nrRecognition;
  double recognitionRate;
};

// Normalization
// Simple normalization which depends only of one bag of words
void bow_simple_normalization(struct svm_problem &svmProblem);

// Gaussian normalization
void get_gaussian_parameters(int k,
			     struct svm_problem svmProblem,
			     double* means,
			     double* stand_devia);
void save_gaussian_parameters(const IMbdd& bdd,
			      double* means,
			      double* stand_devia);
void load_gaussian_parameters(const IMbdd& bdd,
			      double* means,
			      double* stand_devia);
void bow_gaussian_normalization(int k,
				double* means,
				double* stand_devia,
				struct svm_problem &svmProblem);

svm_model **svm_train_ovr(const svm_problem *prob, svm_parameter *param);

double svm_predict_ovr_probs(struct svm_model** models, const svm_node* x, int nbr_class, double* probs,double lamda);

void get_svm_parameter(int k, struct svm_parameter &svmParameter, std::string kernel);

std::vector<double> get_labels_from_prob(const svm_problem *prob);

#endif
