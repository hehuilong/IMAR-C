/** 
 * \file naosvm.cpp
 * \brief Set of functions permiting to import/ predict a svm problem, import/create a svm model
 * \author ROUALDES Fabien (Télécom SudParis)
 * \author HE Huilong (Télécom SudParis)
 * \date 17/07/2013 
*/
#include "naosvm.h" 
#include <math.h>

/**
 * \fn struct svm_problem computeBOW(int label, const KMdata& dataPts, KMfilterCenters& ctrs)
 * \brief Converts the KMdata into a Bag Of Words histogram in the SVM format:
 * label 1:value 2:value 3:value (each lines).
 *
 * \param[in] dataPts The KMdata.
 * \param[in] ctrs The centers.
 * \return The svm problem in a structure.
 */
struct svm_problem computeBOW(int label, const KMdata& dataPts, KMfilterCenters& ctrs){
  int k = ctrs.getK();
  int nPts = dataPts.getNPts();
  
  // 1. Getting assignments 
  KMctrIdxArray closeCtr = new KMctrIdx[dataPts.getNPts()]; // dataPts = 1 label
  double* sqDist = new double[dataPts.getNPts()];
  ctrs.getAssignments(closeCtr, sqDist); 
  
  // 2. Initializing histogram
  float* bowHistogram = NULL;
  bowHistogram = new float[k];
  for(int centre = 0; centre<k; centre++)
    bowHistogram[centre]=0;
  
  // 3. Filling histogram
  for(int point = 0; point < nPts ; point++){
    bowHistogram[closeCtr[point]]++;
  }
  delete[] closeCtr;
  delete[] sqDist;
  
  // 5. Exporting the BOW in the structure svmProblem
  struct svm_problem svmProblem;
  int l=1;
  svmProblem.l = l;
  svmProblem.y = (double*) malloc(svmProblem.l * sizeof(double));
  svmProblem.x = (struct svm_node **) malloc(svmProblem.l * sizeof(struct svm_node *));
  int idActivity = 0;
  while(idActivity < l){
    svmProblem.y[idActivity] = label;
    int notZero = 0;
    int center = 0;
    while(center < k){
      if(bowHistogram[center] != 0)
    	notZero++;
      center++;
    }
    int i = 0;
    svmProblem.x[idActivity] = (svm_node *) malloc((notZero + 1) * sizeof(svm_node));
    center = 0;
    while(center < k){
      if(bowHistogram[center] != .0){
    	svmProblem.x[idActivity][i].index = center + 1;
     	svmProblem.x[idActivity][i].value = bowHistogram[center];
    	i++;
      }
      center++;
    }
    svmProblem.x[idActivity][(notZero-1)+1].index = -1;
    // It is the end of the table we do not need to add a value
    idActivity++;
  }
  delete[] bowHistogram;
  
  return svmProblem; 
}

/**
 * \fn struct svm_model* create_svm_model(std::string bowFile, int k)
 * \brief Create the SVM model of the activities present in a file.
 *
 * \param[in] bowFile The name of the file containing the BOWs.
 * \param[in] k The number of clusters (dimension of a BOW).
 * \return The SVM model.
 */
struct svm_model* create_svm_model(int k, struct svm_problem svmProblem){
  // SVM PARAMETER
  struct svm_parameter svmParameter;
  svmParameter.svm_type = C_SVC;
  svmParameter.kernel_type = RBF;
  //  svm.degree
  svmParameter.gamma = 1.0/k;
  // double coef0;
  
  /* For training only : */
  svmParameter.cache_size = 100; // in MB
  svmParameter.eps = 1e-3; // stopping criteria
  svmParameter.C = 35; // default = 1
  
  // change the penalty for some classes
  svmParameter.nr_weight = 0;
  svmParameter.weight_label = NULL;
  svmParameter.weight = NULL;
    
  //  double nu; // for NU_SVC, ONE_CLASS, and NU_SVR
  //  double p;	// for EPSILON_SVR 
  
  svmParameter.shrinking = 1;	/* use the shrinking heuristics */
  svmParameter.probability = 0; /* do probability estimates */
  
  return svm_train(&svmProblem,&svmParameter);
}
void bow_gaussian_normalization(int k,
				double* means,
				double* stand_devia,
				struct svm_problem &svmProblem
				){
  struct svm_node* node = NULL;
  for(int a=0 ; a<svmProblem.l ; a++){
    node = svmProblem.x[a];
    int i = 0;
    while(node[i].index != -1){
      int index = node[i].index - 1;
      int value = node[i].value;
      node[i].value = (value-means[index])/stand_devia[index];
      i ++;
    }
  }
}

/**
 * the definitions of class MatrixC
 */
MatrixC::MatrixC(int nr_class, int* labels){
  int n = this->num_labels = nr_class;
  this->labels = new double[n];
  for(int i=0; i<n; i++){
    this->labels[i] = labels[i];
  }
  this->m = new int*[n];
  for(int i=0; i<n; i++){
    this->m[i] = new int[n];
  }
  this->m_fre = new double*[n];
  for(int i=0; i<n; i++){
    this->m_fre[i] = new double[n];
  }
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      this->m[i][j] = 0;
      this->m_fre[i][j] = .0;
    }
  }
  
  this->nrTest = 0;
  this->nrRecognition = 0;
}
MatrixC::MatrixC(const svm_model* model){
  int n = this->num_labels = model->nr_class;
  this->labels = new double[n];
  for(int i=0; i<n; i++){
    this->labels[i] = model->label[i];
  }
  this->m = new int*[n];
  for(int i=0; i<n; i++){
    this->m[i] = new int[n];
  }
  this->m_fre = new double*[n];
  for(int i=0; i<n; i++){
    this->m_fre[i] = new double[n];
  }
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      this->m[i][j] = 0;
      this->m_fre[i][j] = .0;
    }
  }
  
  this->nrTest = 0;
  this->nrRecognition = 0;
}
MatrixC::~MatrixC(){
  delete [] this->labels;
  for(int i=0; i<this->num_labels; i++){
    delete [] this->m[i];
    delete [] this->m_fre[i];
  }
  delete [] this->m;
  delete [] this->m_fre;
}
int MatrixC::getIndex(double lab){
  for(int i=0; i<this->num_labels; i++){
    if(this->labels[i] == lab){
      return i;
    }
  }
  return -1;
}
void MatrixC::addTransfer(double lab_in,double lab_out){
  int index_in = this->getIndex(lab_in);
  int index_out = this->getIndex(lab_out);
  this->m[index_in][index_out]++;
  if(lab_in == lab_out) this->nrRecognition++;
  this->nrTest++;
}
void MatrixC::calculFrequence(){
  int num = this->num_labels;
  for(int i=0; i<num; i++){
    int total = 0;
    for(int j=0; j<num; j++){
      total += this->m[i][j];
    }
    for(int j=0; j<num; j++){
      this->m_fre[i][j] = (double)this->m[i][j]/(double)total;
    }
  }
  this->recognitionRate = (double)this->nrRecognition / (double)this->nrTest;  
}

double** MatrixC::getMatrix(){return this->m_fre;}

void MatrixC::output(){
  using namespace std;
  int num = this->num_labels;
  cout<<"===Confusion Matrix==="<<endl;
  cout<<setw(6)<<' ';
  for(int i=0;i<num;i++){
    cout<<setw(6)<<setiosflags(ios::fixed)<<setprecision(0)<<this->labels[i];
  }
  cout<<endl;
  for(int i=0;i<num;i++){
    cout<<setw(6)<<setiosflags(ios::fixed)<<setprecision(0)<<this->labels[i];
    for(int j=0;j<num;j++){
      cout<<setw(6)<<setprecision(2)<<this->m_fre[i][j];
    }
    cout<<endl;
  }
  cout<<"===END Confusion Matrix==="<<endl;
}

void MatrixC::exportMC(std::string folder, std::string file){
  std::string MCfile(folder + "/" + file);
  // Ouverture en écriture avec effacement du fichier ouvert
  ofstream out(MCfile.c_str(), ios::out | ios::trunc);  
  int num = this->num_labels;
  for(int i=0;i<num;i++){
    for(int j=0;j<num;j++){
      out<<setw(7)<< setprecision(2)<< this->m_fre[i][j];
    }
    out<<endl;
  }
  out<<endl<<endl;
  for(int i=0;i<num;i++){
    for(int j=0;j<num;j++){
      out<<setw(7)<< this->m[i][j];
    }
    out<<endl;
  }
}

void bow_simple_normalization(struct svm_problem& svmProblem){
  int nrBows = svmProblem.l;
  int i;
  double sum;
  for(int l=0 ; l<nrBows ; l++){
    i=0;
    
    sum = 0;
    while((svmProblem.x[l])[i].index != -1){
      sum += (svmProblem.x[l])[i].value;
      i++;
    }
    
    i=0;
    while((svmProblem.x[l])[i].index != -1){
      (svmProblem.x[l])[i].value /= sum;
      i++;
    }
  }
}

void destroy_svm_problem(struct svm_problem svmProblem){
  free(svmProblem.y);
  //svmProblem.y = NULL;
  
  for(int a=0 ; a<svmProblem.l ; a++){
    free(svmProblem.x[a]);
    //svmProblem.x[a] = NULL;
  }
  free(svmProblem.x);
  //svmProblem.x = NULL;
}

// Add only one BOW
void addBOW(struct svm_node* bow, double label,
	    struct svm_problem& svmProblem){
  int maxl = 1000;
  int maxdim = 500;
  if(svmProblem.y==NULL || svmProblem.x==NULL){
    //cerr << "Huilong" << endl;
    svmProblem.y=(double*)malloc(maxl*sizeof(double));
    svmProblem.x=(struct svm_node**)malloc(maxl*sizeof(struct svm_node*));
    if(svmProblem.y==NULL || svmProblem.x==NULL){
      std::cerr << "Malloc error of svmProblem.x and svmProblem.y!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  svmProblem.y[svmProblem.l] = label;
  int d=0;
  while(bow[d].index != -1){
    if(d==0&&(svmProblem.x[svmProblem.l]=(struct svm_node*) malloc(maxdim*sizeof(struct svm_node))) == NULL){
      std::cerr << "Malloc error of svmProblem.x[svmProblem.l]!" << std::endl;
      exit(EXIT_FAILURE);
    }
    svmProblem.x[svmProblem.l][d].index = bow[d].index;
    svmProblem.x[svmProblem.l][d].value = bow[d].value;
    d++;
  }
  svmProblem.x[svmProblem.l][d].index = -1;
  svmProblem.l++;
}

void get_gaussian_parameters(int k,
			     struct svm_problem svmProblem,
			     double* means,
			     double* stand_devia){
  struct svm_node** nodes = svmProblem.x;
  //double* labels = svmProblem.y;
  int num_nodes = svmProblem.l;
  int pointers[num_nodes];
  for(int i=0; i<num_nodes; i++)
    pointers[i] = 0;
  for(int i=0; i<k; i++){
    double components[num_nodes]; 
    double total = 0;
    for(int j=0; j<num_nodes; j++){
      struct svm_node* node = nodes[j];
      components[j] = 0;
      int pointer = pointers[j];
      int index = node[pointer].index;
      if(i+1 == index){
        components[j] = node[pointer].value;
        pointers[j] ++;
      }
      total += components[j];
    }
    means[i] = total/num_nodes;
    double var = 0;
    for(int j=0; j<num_nodes; j++){
      var += (components[j]-means[i])*(components[j]-means[i]);
    }
    var /= num_nodes;
    stand_devia[i] = sqrt(var);  
  }
}
void save_gaussian_parameters(const IMbdd& bdd,
			      double* means,
			      double* stand_devia){
  // Save the gaussian parameters
  std::string meansPath(bdd.getFolder() + "/" + bdd.getMeansFile());
  std::string standPath(bdd.getFolder() + "/" + bdd.getStandardDeviationFile());
  std::cout << meansPath << std::endl << standPath << std::endl;
  std::ofstream outmean(meansPath.c_str());
  std::ofstream outstand(standPath.c_str());
  //std::cout << meansPath << " " << standPath << std::endl;
  for(int i=0 ; i<bdd.getK() ; i++){
    outmean<<means[i]<<std::endl;
    outstand<<stand_devia[i]<<std::endl;
  }
}

void load_gaussian_parameters(const IMbdd& bdd, 
			      double* means,
			      double* stand_devia){
  std::string meansPath(bdd.getFolder() + "/" + bdd.getMeansFile());
  std::string standPath(bdd.getFolder() + "/" + bdd.getStandardDeviationFile());
  std::ifstream inmean(meansPath.c_str());
  std::ifstream instand(standPath.c_str());
  std::cout << meansPath << " " << standPath << std::endl;
  for(int i=0; i<bdd.getK() ; i++){
    inmean >> (means)[i];
    instand >> (stand_devia)[i];
  }
}

// svm training funciton using the strategy one-vs-rest
svm_model **svm_train_ovr(const svm_problem *prob, svm_parameter *param){
  int nr_class;
  vector<double> label;
  label = get_labels_from_prob(prob);
  nr_class = label.size();
  int l = prob->l;
  svm_model **model = new svm_model*[nr_class];
  if(nr_class == 1){
    std::cerr<<"Training data in only one class. Aborting!"<<std::endl;
    exit(EXIT_FAILURE);
  }
  double *temp_y = new double[l];
  param->weight = new double[2];
  param->weight_label = new int[2];
  for(int i=0;i<nr_class;i++){
    double label_class = label[i];
    for(int j=0;j<l;j++) temp_y[j] = prob->y[j];
    param->nr_weight = 2;
    param->weight_label[1] = ceil(label_class);
    param->weight_label[0] = 0;
    param->weight[0] = param->weight[1] = 0;
    for(int j=0;j<l;j++){
      if(label_class != prob->y[j]){
        prob->y[j] = 0;
        param->weight[0] += 1;
      }
      else{
        param->weight[1] += 1;
      }
    }
    param->weight[0] = 1/sqrt(param->weight[0]);
    param->weight[1] = 1/sqrt(param->weight[1]);
    model[i] = svm_train(prob,param);
    for(int j=0;j<l;j++) prob->y[j] =  temp_y[j];
  }
  delete [] temp_y;
  delete [] param->weight;
  delete [] param->weight_label;
  return model;
}

// svm predictor using the one-vs-rest strategy returning the probilities
double svm_predict_ovr_probs(svm_model** models, const svm_node* x,int nbr_class, double* probs, double lamda){
  if(lamda <= 0){
    std::cerr<<"ERROR: svm_predict_ovr_probs(): lamda must be positive!"<<std::endl;
    exit(EXIT_FAILURE);
  }
  double *decvs = new double[nbr_class];
  double *labels = new double[nbr_class];
  double totalExpDecv = 0;
  for(int i=0;i<nbr_class;i++){
    labels[i] = models[i]->label[0]+models[i]->label[1];
    double label_pred = svm_predict_values(models[i],x,&(decvs[i])); 

    if(decvs[i]<0 && label_pred>0)
      decvs[i] = -decvs[i];
    if(decvs[i]>0 && label_pred<=0)
      decvs[i] = -decvs[i];
    
    probs[i] = exp(lamda*decvs[i]);
    totalExpDecv += probs[i];
  }
  for(int i=0;i<nbr_class;i++){
    probs[i] = probs[i]/totalExpDecv;
  }
  double maxvalue = decvs[0];
  double label = labels[0];
  for(int i=1;i<nbr_class;i++){
    if(decvs[i]>maxvalue){
      maxvalue = decvs[i];
      label = labels[i];
    }
  }
  delete [] decvs;
  delete [] labels;
  return label;
}

//print ovr label:prob
void svm_ovr_print(double *labels, double *probs, int nbr_class){
  using namespace std;
  cerr<<"Ovr labels and probabilities:"<<endl;
  for(int i=0;i<nbr_class;i++){
    cerr<<setiosflags(ios::fixed)<<labels[i]<<":"<<probs[i]<<" ";
  }
  cerr<<endl;
  return;
}

void get_svm_parameter(int k,struct svm_parameter &svmParameter, std::string kernel){
  const char* kernel_type[]=
  {
  	"linear","polynomial","rbf","sigmoid","chis","rbfchis","inters","precomputed",NULL
  };
  int i;
  for(i=0;kernel_type[i];i++){
    if(strcmp(kernel_type[i],kernel.c_str())==0){
      svmParameter.kernel_type=i;
      break;
    }
  }
  if(kernel_type[i] == NULL){
    std::cerr << "Unexpected SVM kernel, using chi-square kernel..." << std::endl;
    svmParameter.kernel_type=CHIS;
  }
  if(kernel == "rbfchis"){
    // Find a better A by cross validation 
    svmParameter.rbfchisA = 230;
  }
  svmParameter.svm_type = C_SVC;
  svmParameter.gamma = 1.0/k;
  svmParameter.cache_size = 100; // in MB
  svmParameter.eps = 1e-3; // stopping criteria
  svmParameter.C = 1;
  // Change the penalty for some classes
  svmParameter.nr_weight = 0;
  svmParameter.weight_label = NULL;
  svmParameter.weight = NULL;
  // Use the shrinking heuristics 
  svmParameter.shrinking = 1;	
  // Probability estimates
  svmParameter.probability = 0;
}

std::vector<double> get_labels_from_prob(const svm_problem *prob){
  int l = prob->l;
  double *y = prob->y;
  vector<double> labels;
  for(int i=0; i<l; i++){
    bool exist = false;
    for(std::vector<double>::iterator itr = labels.begin(); itr != labels.end();itr++){
      if(y[i] == *itr){
        exist = true;
        break;
      }
    }
    if(!exist){
      labels.push_back(y[i]);
    }
  }
  return labels;
}
