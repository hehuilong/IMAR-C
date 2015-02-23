/**
 * \file naomngt.cpp
 * \brief Set of functions permiting to manage the activity recognition BDD of Bag Of Words.
 * \author ROUALDES Fabien (Télécom SudParis)
 * \author HUILONG He (Télécom SudParis)
 * \date 25/07/2013
 */
#include "naomngt.h"
#include <time.h>
#include <algorithm>

/**
 * \fn std::string inttostring(int int2str)
 * \brief Converts an int into a string.
 * \param[in] int2str The int to convert.
 * \return The string converted.
 */
std::string inttostring(int int2str){
  std::ostringstream oss;
  oss << int2str;
  std::string result = oss.str(); 
  return result;
}

/**
 * \fn void refreshBdd(std::string bddName, int desc, int maxPts)
 * \brief Deletes all files excepted videos and extracts STIPs again.
 *
 * \param[in] bddName The name of the BDD containing videos.
 * \param[in] dim The STIPs dimension.
 * \param[in] maxPts The maximum number of STIPs we can extract.
 */
void im_refresh_bdd(std::string bddName,
		    int scale_num,
		    std::string descriptor){
  std::string path2bdd("bdd/" + bddName);
  IMbdd bdd = IMbdd(bddName,path2bdd);
  bdd.load_bdd_configuration(path2bdd,"imconfig.xml");
  // Saving the new new descriptor with its dimension
  int dim = getDim(descriptor);
  bdd.changeDenseTrackSettings(scale_num,
			       descriptor,
			       dim);
  bdd.write_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
  // Deleting testing files and training files 
  DIR* repBDD = opendir(path2bdd.c_str());
  if (repBDD == NULL){
    std::cerr << "Impossible top open the BDD directory"<< std::endl;
    exit(EXIT_FAILURE);
  }
  struct dirent *ent;
  while ( (ent = readdir(repBDD)) != NULL){
    std::string file = ent->d_name;
    std::string f = path2bdd + "/" + file;
    if((file.compare("concatenate.fp.test") == 0) || (file.compare("concatenate.fp.train") == 0) ||
       (file.compare("concatenate.bow.test") == 0) || (file.compare("concatenate.bow.train") == 0) || 
       (file.compare("files.test") == 0) || (file.compare("files.train") == 0)){
      std::cout << "Deleting " << f << std::endl;
      remove(f.c_str());
    }
  }
  closedir(repBDD);
  // Get training people
  std::vector<std::string> people = bdd.getPeople();
  std::vector<std::string> testPeople = bdd.getTestPeople();
  // Combine all the people
  for(vector<string>::iterator person = testPeople.begin() ; person != testPeople.end() ; ++person){
    people.push_back(*person);
  }
  vector<string> activities = bdd.getActivities();
  int num_act = activities.size();
  // Array of number of feature points of each activity
  int* num_points = new int[num_act];
  // Array of number of videos of each activity
  int* num_videos = new int[num_act];
  for(int i=0 ; i<num_act ; i++){
    num_points[i] = 0;
    num_videos[i] = 0;
  }
  // Compute the feature points of all the video
  for(vector<string>::iterator person = people.begin() ; person != people.end() ; ++person){
    std::string personFolder(path2bdd + "/" + *person);
    im_refresh_folder(bdd, personFolder, num_points, num_videos);
  }
  cout << "==DenseTrack done==" << std::endl;
  cout << "Activity: <num of feature points>/<num of videos> = <points per video>" << std::endl;
  int total_points = 0;
  int total_videos = 0;
  for(unsigned int i=0 ; i<activities.size() ; i++){
    total_points += num_points[i];
    total_videos += num_videos[i];
    cout << activities[i] << ": " << num_points[i] << " / " 
         << num_videos[i] << " = " << setiosflags(ios::fixed) << setprecision(2) << num_points[i]*1.0/num_videos[i] 
         << endl;
  }
  cout << "Total number of points: " << total_points << endl;
  cout << "Total number of videos: " << total_videos << endl;
  im_concatenate_bdd_feature_points(bdd.getFolder(),
            people,
				    bdd.getActivities());
  delete[] num_points;
}

/**
 * \fn void im_refresh_folder(std::string folder, std::vector<std::string> activities, int scale_num, int dim, int maxPts)
 * \brief Deletes all files excepted videos and extracts STIPs again.
 *
 * \param[in] folder the path to the folder containing videos.
 * \param[in] activities the vector containing all the activities.
 * \param[in] scale_num the number of scales used for the feature points extraction.
 * \param[in] dim the dimension of the feature points.
 * \param[in] maxPts the maximum number of feature points we can extract.
 */
void im_refresh_folder(const IMbdd& bdd, std::string folder, int* num_points, int* num_videos){
  std::vector<std::string> activities = bdd.getActivities();
  int scale_num = bdd.getScaleNum();
  std::string descriptor(bdd.getDescriptor());
  int dim = bdd.getDim();
  int maxPts = bdd.getMaxPts();
  std::cerr << "scale num: " << scale_num << " desc: " <<descriptor << " dim : "<< dim << " maxpts: "<<maxPts<<std::endl;
  
  // Deleting all features points
  for(std::vector<std::string>::iterator activity = activities.begin() ;
      activity != activities.end() ;
      ++activity){
    string rep(folder + "/" + *activity + "/fp");
    DIR * repertoire = opendir(rep.c_str());
    if (repertoire == NULL){
      std::cerr << "Impossible to open the fp folder!" << std::endl;
      exit(EXIT_FAILURE);
    }
    struct dirent *ent;
    while ( (ent = readdir(repertoire)) != NULL){
      if(strcmp(ent->d_name,".") != 0 && strcmp(ent->d_name,"..") != 0){
        std::string filename(ent->d_name);
        filename = rep+"/"+filename;
        std::cout << "Deleting " << filename << std::endl;
	      remove(filename.c_str());
      }
    }
    closedir(repertoire);
  } 
  int currentAct = -1; 
  // Extracting feature points for each videos
  for(std::vector<std::string>::iterator activity = activities.begin() ;
      activity != activities.end() ;
      ++activity){
    ++currentAct;
    std::string avipath(folder + "/" + *activity + "/avi");
    std::cerr <<"Entering directory: "<< avipath<< std::endl;
    std::string FPpath(folder + "/" +  *activity + "/fp");
    DIR * repertoire = opendir(avipath.c_str());
    if (repertoire == NULL){
      std::cerr << "Impossible to open the avi folder for extraction!" << std::endl;
      exit(EXIT_FAILURE);
    }
    struct dirent * ent;
    int j = 1;
    while ( (ent = readdir(repertoire)) != NULL){
      std::string file = ent->d_name;
      if(file.compare(".") != 0 && file.compare("..") != 0){
	      string idFile = inttostring(j);
        // Extract feature points from the videos and save them in the repertory /path/to/folder/activity/fp
        KMdata dataPts(dim,maxPts);
        string videoInput(avipath + "/" + file);
        string stipOutput(FPpath + "/" + *activity + "-" + idFile + ".fp");
	      int nPts;
        long starttime, finishtime;
        double duration;

        std::cerr<<"Extracting: "<<videoInput<<std::endl;
        num_videos[currentAct] ++;
        starttime = clock();
	      nPts = extract_feature_points(videoInput,
	      			      scale_num, descriptor, dim,
	      			      maxPts, dataPts);		
        finishtime = clock();
        num_points[currentAct] += nPts;
        duration = finishtime - starttime;
        duration /= CLOCKS_PER_SEC;
        std::cerr<<"["<<duration<<" seconds]"<<std::endl;
	      if(nPts != 0){
	        dataPts.setNPts(nPts);
	        exportSTIPs(stipOutput, dim, dataPts);
	      }
        else{
          std::cout << "No feature point detected" << std::endl;
        }
	      j++;	
      }
    }
    closedir(repertoire);
    // The extraction of the videos feature points of the activity is terminated.
  }
}


void im_concatenate_bdd_feature_points(std::string path2bdd,
				       std::vector<std::string> people,
				       std::vector<std::string> activities){
  for(std::vector<std::string>::iterator person = people.begin() ;
      person != people.end() ;
      ++person)
    im_concatenate_folder_feature_points(path2bdd + "/" + *person, activities);  
}

// Concatenate the feature points of each activities of a folder
void im_concatenate_folder_feature_points(std::string folder,
					  std::vector<std::string> activities){
  // First we delete the file if it exists
  for(std::vector<std::string>::iterator activity = activities.begin() ; activity != activities.end() ; ++activity){
    std::string rep(folder + "/" + *activity);
    std::string path2concatenate(rep + "/concatenate." + *activity + ".fp");
    // We open the directory folder/label
    DIR * directory = opendir(rep.c_str());
    if (directory == NULL){
      std::cerr << "Impossible to open the folder " << rep <<" for deletion!"<< std::endl;
      exit(EXIT_FAILURE);
    }
    struct dirent *ent;
    while ( (ent = readdir(directory)) != NULL){
      std::string file(ent->d_name);
      if(file.compare("concatenate." + *activity + ".fp") == 0)
	      remove(path2concatenate.c_str());
    }
    closedir(directory);
  }
  
  // Then we concatenate the feature points per activities (for KMeans)
  for(std::vector<std::string>::iterator activity = activities.begin() ;
      activity != activities.end() ;
      ++activity){
    string rep(folder + "/" + *activity + "/fp");
    DIR * directory = opendir(rep.c_str());
    if (directory == NULL){
      std::cerr << "Impossible to open the feature points folder for concatenation!" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::string activityOutPath(folder + "/" + *activity + "/concatenate." + *activity + ".fp");
    ofstream activityOut(activityOutPath.c_str());
    struct dirent * ent;
    while ( (ent = readdir(directory)) != NULL){
      std::string file = ent->d_name;
      if(file.compare(".") != 0 && file.compare("..") != 0){
	      std::string path2fp(rep + "/" + file);
	      ifstream in(path2fp.c_str());
	      std::string line;
        while (std::getline(in, line)){
	        activityOut << line << std::endl;
	      }
      }
    }
    closedir(directory);
  }
}					

int im_create_normal_training_means(IMbdd bdd,
				       const std::vector<std::string>& trainingPeople 
				       ){
  std::string path2bdd(bdd.getFolder());
  int dim = bdd.getDim();
  int maxPts = bdd.getMaxPts();
  
  std::vector <std::string> activities = bdd.getActivities();
  int nr_class = activities.size();
  
  int nr_people = trainingPeople.size();
  
  // The total number of centers
  int k = bdd.getK();
  // vDataPts for saving all the feature points
  double **vDataPts = new double *[maxPts];
  // Number of feature points for each class
  int *nrFP = new int[nr_class]; 
  // Total number of feature points
  int totalNrFp = 0; 
  int currentActivity = 0;
  // For each activity
  for(std::vector<std::string>::iterator activity = activities.begin() ;
      activity != activities.end() ;
      ++activity){
    // We concatenate all the training people
    nrFP[currentActivity] = 0;
    // Number of feature points per person   
    int nrFPpP[nr_people]; 
    double*** activityDataPts = new double**[nr_people];
    int currentPerson = 0;
    for(std::vector<std::string>::const_iterator person = trainingPeople.begin() ;
	      person != trainingPeople.end() ;
	      ++person){
      nrFPpP[currentPerson] = 0;
      std::string rep(path2bdd + "/" + *person + "/" + *activity);
      DIR * repertoire = opendir(rep.c_str());
      if (!repertoire){
	      std::cerr << "Impossible to open the feature points directory!" << std::endl;
	      exit(EXIT_FAILURE);
      }
      // Checking that the file concatenate.<activity>.fp exists
      struct dirent * ent = readdir(repertoire);
      std::string file(ent->d_name);
      while (ent && (file.compare("concatenate." + *activity + ".fp")) != 0){
	      ent = readdir(repertoire);
	      file = ent->d_name;
      }
      if(!ent){
	      std::cerr << "Cannot find concatenate.<activity>.fp" << std::endl;
	      exit(EXIT_FAILURE);
      }
      // Importing the feature points
      std::string path2FP(rep + "/" + file);
      KMdata kmData(dim,maxPts);
      nrFPpP[currentPerson] = importSTIPs(path2FP, dim, maxPts, &kmData);
      if(nrFPpP[currentPerson] != 0){
	      activityDataPts[currentPerson] = new double*[nrFPpP[currentPerson]];
	      for(int n = 0 ; n<nrFPpP[currentPerson] ; n++){
	        activityDataPts[currentPerson][n] = new double [dim];
	        for(int d = 0 ; d<dim ; d++)
	          activityDataPts[currentPerson][n][d] = kmData[n][d];
	      }
      } 
      // Else the current person does not participate in this activity
      nrFP[currentActivity] += nrFPpP[currentPerson];
      currentPerson++;
    } // End for person
    // Saving feature points in vDataPts
    int index=0;
    for(int p=0 ; p<nr_people ; p++){
      for(int fp=0 ; fp<nrFPpP[p] ; fp++){
	      vDataPts[index + totalNrFp] = new double[dim];
	      for(int d=0 ; d<dim ; d++){
	        vDataPts[index + totalNrFp][d] = activityDataPts[p][fp][d];
	      }
	      index++;
      }
    } 
    totalNrFp += nrFP[currentActivity];
    // Delete activityDataPts
    for(int p=0 ; p<nr_people ; p++){
      if(nrFPpP[p] != 0){
        for(int fp=0 ; fp < nrFPpP[p] ; fp++)
	        delete[] activityDataPts[p][fp];
        delete[] activityDataPts[p];
      }
    }
    delete[] activityDataPts;
    currentActivity++;
  } // end for activity
  
  // Total number of feature points
  int ttFP = 0;
  for(int a=0 ; a<nr_class ; a++){
    ttFP += nrFP[a];
  }
  if(ttFP != totalNrFp) 
    std::cerr << "!!!!!!!!WRONG!!!!!!!" << std::endl;
  // KMeans algorithm on all the points 
  // The iteration coefficient (Ivan's algorithm)
  int ic = 3; 
  // Copy the feature points to KMdata object
  KMdata kmData(dim,ttFP);
  for(int n=0 ; n<ttFP; n++)
    for(int d=0 ; d<dim ; d++)
	    kmData[n][d] = vDataPts[n][d];
  // Delete vDataPts
  for(int n=0 ; n<ttFP ; n++)
    delete [] vDataPts[n];
  delete [] vDataPts;
  // Run kmeans on kmData
  kmData.setNPts(ttFP);
  kmData.buildKcTree();
  KMfilterCenters kmCtrs(k,kmData);
  kmIvanAlgorithm(ic, dim, kmData,k, kmCtrs);
  exportCenters(bdd.getFolder() + "/" + bdd.getKMeansFile(),
		dim, k, kmCtrs);
  delete[] nrFP; 
  std::cout << "KMeans done, nomber of points: " << ttFP << std::endl;
  return k;
}

int im_create_specifics_training_means(IMbdd bdd,
				       const std::vector<std::string>& trainingPeople 
				       ){
  std::string path2bdd(bdd.getFolder());
  int dim = bdd.getDim();
  int maxPts = bdd.getMaxPts();
  std::vector <std::string> activities = bdd.getActivities();
  int nr_class = activities.size();
  
  int nr_people = trainingPeople.size();
  // The total number of centers
  int k = bdd.getK();
  int subK = k/nr_class;
  if(k%nr_class != 0){
    std::cerr << "K is no divisible by the number of activities !!" << std::endl;
    exit(EXIT_FAILURE);
  }
  double ***vDataPts = new double**[nr_class];
  int nrFP[nr_class]; // number of feature points for each class
  int currentActivity = 0;
  // For each activity
  for(std::vector<std::string>::iterator activity = activities.begin() ;
      activity != activities.end() ;
      ++activity){
    nrFP[currentActivity] = 0;
    // We concatenate all the training people
    int nrFPpP[nr_people]; // number of feature points per person
    double*** activityDataPts = new double**[nr_people];
    int currentPerson = 0;
    for(std::vector<std::string>::const_iterator person = trainingPeople.begin() ;
	      person != trainingPeople.end() ;
	      ++person){
      nrFPpP[currentPerson] = 0;
      std::string rep(path2bdd + "/" + *person + "/" + *activity);
      DIR * repertoire = opendir(rep.c_str());
      if (!repertoire){
	      std::cerr << "Impossible to open the feature points directory!" << std::endl;
	      exit(EXIT_FAILURE);
      }
      // Checking that the file concatenate.<activity>.fp exists
      struct dirent * ent = readdir(repertoire);
      std::string file(ent->d_name);
      while (ent && (file.compare("concatenate." + *activity + ".fp")) != 0){
	      ent = readdir(repertoire);
	      file = ent->d_name;
      }
      if(!ent){
	      std::cerr << "No file concatenate.<activity>.fp" << std::endl;
	      exit(EXIT_FAILURE);
      }
      // Importing the feature points
      std::string path2FP(rep + "/" + file);
      KMdata kmData(dim,maxPts);
      nrFPpP[currentPerson] = importSTIPs(path2FP, dim, maxPts, &kmData);
      if(nrFPpP[currentPerson] != 0){
	      activityDataPts[currentPerson] = new double*[nrFPpP[currentPerson]];
	      for(int n = 0 ; n<nrFPpP[currentPerson] ; n++){
	        activityDataPts[currentPerson][n] = new double [dim];
	        for(int d = 0 ; d<dim ; d++)
	          activityDataPts[currentPerson][n][d] = kmData[n][d];
	      }
      } // else the current person does not participate in this activity
      nrFP[currentActivity] += nrFPpP[currentPerson];
      currentPerson++;
    } // ++person
    
    // Saving people in vDataPts
    vDataPts[currentActivity] = new double*[nrFP[currentActivity]];
    int index=0;
    for(int p=0 ; p<nr_people ; p++){
      for(int fp=0 ; fp<nrFPpP[p] ; fp++){
	      vDataPts[currentActivity][index] = new double[dim];
	      for(int d=0 ; d<dim ; d++){
	        vDataPts[currentActivity][index][d] = activityDataPts[p][fp][d];
	      }
	      index++;
      }
    } // index must be equal to nrFP[currentActivity] - 1
    // Deleting activityDataPts
    for(int p=0 ; p<nr_people ; p++){
      if(nrFPpP[p] != 0){
        for(int fp=0 ; fp < nrFPpP[p] ; fp++)
	        delete[] activityDataPts[p][fp];
        delete[] activityDataPts[p];
      }
    }
    delete[] activityDataPts;
    currentActivity++;
  } // ++activity
  
  // Total number of feature points
  int ttFP = 0;
  for(int a=0 ; a<nr_class ; a++){
    ttFP += nrFP[a];
  }
  
  // Memory allocation of the centers
  double** vCtrs = new double*[k];
  for(int i=0 ; i<k ; i++){
    vCtrs[i] = new double[dim];
  }
  
  // Doing the KMeans algorithm for each activities
  int ic = 3; // the iteration coefficient (Ivan's algorithm)
  int currCenter = 0;
  for(int i=0 ; i<nr_class ; i++){
    KMdata kmData(dim,nrFP[i]);
    for(int n=0 ; n<nrFP[i]; n++)
      for(int d=0 ; d<dim ; d++)
	      kmData[n][d] = vDataPts[i][n][d];
    kmData.setNPts(nrFP[i]);
    kmData.buildKcTree();
    KMfilterCenters kmCtrs(subK,kmData);
    kmIvanAlgorithm(ic, dim, kmData, subK, kmCtrs);
    for(int n=0 ; n<subK ; n++){
      for(int d=0 ; d<dim ; d++){
	      vCtrs[currCenter][d] = kmCtrs[n][d];
      }
      currCenter++;
    }
  }
  
  std::cout << "Concatenate KMdata" << std::endl;
  // Concatenate all the KMdata
  /* it is not necessary
     but the objectif KMfilterCenters must be initialized with
     the KMdata */
  KMdata dataPts(dim,ttFP);
  int nPts = 0;
  for (int i=0 ; i<nr_class ; i++){
    for(int n=0 ; n<nrFP[i]; n++){
      for(int d=0 ; d<dim ; d++){
      	dataPts[nPts][d] = vDataPts[i][n][d];
      }
      nPts++;
    }
  }
  // Releasing vDataPts
  for(int i=0 ; i<nr_class ; i++){
    for(int n=0 ; n<nrFP[i] ; n++)
      delete [] vDataPts[i][n];
    delete [] vDataPts[i];
  }
  delete[] vDataPts;
  
  dataPts.buildKcTree();
  // Returning the true centers
  KMfilterCenters ctrs(k,dataPts);
  for(int n=0 ; n<k ; n++){
    for(int d=0 ; d<dim ; d++){
      ctrs[n][d] = vCtrs[n][d];
    }
  }
  
  // Releasing vCtrs
  for(int i=0 ; i<k ; i++)
    delete [] vCtrs[i];
  delete[]vCtrs;
  
  exportCenters(bdd.getFolder() + "/" + bdd.getKMeansFile(),
		            dim, k, ctrs);
  
  std::cout << "KMeans done, nomber of points: " << ttFP << std::endl;
  return k;
}

void im_kmeans_bdd(std::string bddName, int k, std::string kmeansType){
  std::string path2bdd("bdd/" + bddName);
  std::string KMeansFile(path2bdd + "/" + "training.means");
  
  std::cout <<"Entering database: "<< path2bdd << std::endl;
  
  // Loading BDD
  IMbdd bdd(bddName,path2bdd);
  bdd.load_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
  
  std::vector<std::string> trainingPeople = bdd.getPeople();
  std::vector<std::string> testingPeople = bdd.getTestPeople();
  std::vector<std::string> kmeansPeople(trainingPeople.size()+testingPeople.size());

  std::merge(trainingPeople.begin(),trainingPeople.end(),
            testingPeople.begin(),testingPeople.end(),
            kmeansPeople.begin());
  
  // Saving KMeans settings
  if(kmeansType == "special"){
    bdd.changeKMSettings("special",
	  	       k,
	  	       "training.means");
    im_create_specifics_training_means(bdd, trainingPeople);
  }
  else if(kmeansType == "normal"){
    bdd.changeKMSettings("normal",
	  	       k,
	  	       "training.means");
    im_create_normal_training_means(bdd, trainingPeople);
  }
  else{
    std::cerr << "Unknown KMeans Type!" << std::endl;
    exit(EXIT_FAILURE);
  }
  bdd.write_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
}

/**
 * \fn void im_train_bdd(std::string bddName, int dim, int maxPts, int k)
 * \brief Trains the specified BDD.
 *
 * \param[in] bddName The name of the BDD.
 * \param[in] dim The dimension of the STIPs.
 * \param[in] maxPts The maximum number of points we want to compute.
 * \param[in] k The number of cluster (means).
 */
void im_train_bdd(std::string bddName, std::string kernel, std::string normalization){
  std::string path2bdd("bdd/" + bddName);
  std::string KMeansFile(path2bdd + "/" + "training.means");
  
  std::cout <<"Entering database: "<< path2bdd << std::endl;
  
  // Loading BDD
  IMbdd bdd(bddName,path2bdd);
  bdd.load_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
  
  std::vector<std::string> trainingPeople = bdd.getPeople();
  std::vector<std::string> testingPeople = bdd.getTestPeople();

  // Loading activities
  std::vector<std::string> activities = bdd.getActivities();
  int nrActivities = activities.size();
  int labels[nrActivities];
  int index = 0;
  for(std::vector<std::string>::iterator activity = activities.begin();
      activity != activities.end();
      ++activity){
    labels[index] = index + 1;
    index++;
  }

  // SVM train
  MatrixC trainMC = MatrixC(nrActivities,labels);
  MatrixC testMC = MatrixC(nrActivities, labels);
  double crossValidationAccuracy = im_svm_train(bdd,
						trainingPeople, trainMC,
						testingPeople, testMC, kernel, normalization);
  trainMC.calculFrequence();
  testMC.calculFrequence();
  trainMC.exportMC(path2bdd,"training_confusion_matrix.txt");
  testMC.exportMC(path2bdd,"testing_confusion_matrix.txt");
  bdd.write_bdd_configuration(path2bdd.c_str(),"imconfig.xml");
  
  std::cout << "Resume of the training phase:" << std::endl;
  bdd.show_bdd_configuration();
  std::cout << "Train recognition rate:" << std::endl;
  std::cout << "\t cross validation accuracy=" << crossValidationAccuracy << std::endl;
  std::cout << "\t tau_train=" << trainMC.recognitionRate*100 << "%" << std::endl;
  std::cout << "\t tau_test=" << testMC.recognitionRate*100 << "%" << std::endl;
  std::cout << "\t total number of train BOWs=" << trainMC.nrTest << std::endl;
  std::cout << "\t total number of test BOWs=" << testMC.nrTest << std::endl;
}

double im_training_leave_one_out(const IMbdd& bdd,
				 const std::vector<std::string>& trainingPeople,
				 const std::map <std::string, struct svm_problem>& peopleBOW,
				 int& minC, int& maxC,
				 int& minG, int& maxG,
				 struct svm_parameter& svmParameter){
  if(trainingPeople.size() < 2){
    std::cerr << "Impossible to make a leave-one-out-cross-validation with less than 2 people!" << std::endl;
    exit(EXIT_FAILURE);
  }
  int nrActivities = bdd.getActivities().size();
  std::string path2bdd(bdd.getFolder());
  int C;
  int bestC;
  double bestAccuracy = -1;
  for(C=minC; C<=maxC ; C++){
    //for(gamma = minG ; gamma <= maxG ; gamma++){
    svmParameter.C = pow(2,C);
    std::cout << std::endl << "C: " << svmParameter.C << std::endl;
    //svmParameter.gamma = pow(2,gamma);
    double XValidationAccuracy = 0;
    // For each couple (C,gamma) we do people.size() leave one outs
    for(std::vector<std::string>::const_iterator person = trainingPeople.begin();
	  person != trainingPeople.end();
	  ++person){
      int total_correct = 0;
      int nrBOW = 0;
	    // This person will be the testing person
	    
	    // We do the training with the others
	    struct svm_problem trainingProblem;
	    trainingProblem.l = 0;
	    trainingProblem.x = NULL;
	    trainingProblem.y = NULL;
	    for(std::vector<std::string>::const_iterator trainingPerson = trainingPeople.begin();
	        trainingPerson != trainingPeople.end();
	        ++ trainingPerson){
	      // Training with the other person
	      if((*trainingPerson).compare(*person) != 0){
	        // For each person we train the rest and we test the person
	        // Then compute the mean of the cross validation accuracy 
	        for(int i=0 ; i < peopleBOW.at(*trainingPerson).l ; i++){
	          struct svm_node* bow = peopleBOW.at(*trainingPerson).x[i];
	          addBOW(bow, peopleBOW.at(*trainingPerson).y[i],trainingProblem);
	        }
	      }
	    }
	    struct svm_model** svmModels = svm_train_ovr(&trainingProblem,&svmParameter);
	
	    // Making test
	    for(int i=0 ; i<peopleBOW.at(*person).l ; i++){
	      double* probs = new double[nrActivities];
	      double lab_in = peopleBOW.at(*person).y[i];
	      double lab_out = svm_predict_ovr_probs(svmModels,peopleBOW.at(*person).x[i],
	    					 nrActivities,probs,2);
	      delete []probs;
	      if(lab_in == lab_out)
	        total_correct++;
	      nrBOW++;
	    }
      double accur = total_correct*1.0/nrBOW; 
      XValidationAccuracy += accur;
      //std::cout << "Leaving " << (*person) << " out. Accuracy: " << accur << std::endl;
	    // Releasing svmModels memory
	    for(int i=0 ; i<nrActivities ; i++){
	      svm_free_and_destroy_model(&svmModels[i]);
	    }
	    delete[] svmModels;
	    destroy_svm_problem(trainingProblem);
    } // leave one out

    XValidationAccuracy /= trainingPeople.size();
    if(XValidationAccuracy > bestAccuracy){
	    bestAccuracy = XValidationAccuracy;
	    bestC = C;
	    //bestG = gamma;
    }
    std::cout << "Mean Accuracy: " << XValidationAccuracy << std::endl;
    // } // gamma loop 
  } // C loop
  svmParameter.C = pow(2,bestC);
  //svmParameter.gamma = pow(2,bestG);
  std::cout << "Best C: " << svmParameter.C << std::endl; 
  return bestAccuracy;
}

double im_svm_train(IMbdd& bdd,
		    const std::vector<std::string>& trainingPeople,
		    MatrixC& trainMC,
		    const std::vector<std::string>& testingPeople,
		    MatrixC& testMC, 
        std::string kernel, std::string normalization){
  std::string path2bdd(bdd.getFolder());
  int k = bdd.getK();
  // Computing Bag Of Words for each person 
  std::cout << "Computing BOWs..." << std::endl;
  std::map<std::string, struct svm_problem> peopleBOW;
  std::map<std::string, int> activitiesLabel;
  std::cout << "Computing the SVM problem (ie. Bag of Words) of all the persons..." << std::endl;
  im_compute_bdd_bow(bdd,peopleBOW);
  // All the BOW are saved in peopleBOW
  // Normalizing the Bags of Words (bags of features) 
  std::cout << "Normalizing the BOW..." << std::endl; 
  if(normalization != "simple" && normalization != "gaussian" && normalization != "non" && normalization != "both"){
    std::cerr << "Unexpected normalization type, Bags of Words not normalized" << std::endl;
    normalization = "non";
  }
  bdd.changeNormalizationSettings(normalization,
				  "means.txt",
				  "stand_devia.txt");
  // In case of non-gaussian normalization, these two txt files above are not used
  if(normalization == "gaussian"){
    // Calculate the gaussian means and stand deviations of each component of training people BoWs
    im_compute_gaussian_param(bdd,trainingPeople,peopleBOW); 
  }
  if(normalization == "simple" || normalization == "gaussian" || normalization == "both"){
    std::vector<std::string> allPeople(trainingPeople);
    for(std::vector<std::string>::const_iterator person = testingPeople.begin();
        person != testingPeople.end();
        ++person){
      allPeople.push_back(*person);  
    }
    im_normalize_bdd_bow(bdd,allPeople,peopleBOW); 
  }
  // * SVM *
  // (One Versus the Rest (OVR))
  // SVM parameters
  struct svm_parameter svmParameter;
  get_svm_parameter(k,svmParameter,kernel);
  if(kernel == "rbfchis"){
    std::cout << "Computing parameter A ..." << std::endl;
    compute_rbfchis_A(trainingPeople,svmParameter,peopleBOW);
    std::cout << "A = " << svmParameter.rbfchisA << std::endl;
  }
  std::cout << "Searching the best C ..." << std::endl;
  // We have to do a leave one out with the training data
  // For the moment we use the same training centers to find Gamma and C
  int minC = -10, maxC = 10;
  int minG = -10, maxG = 3;
  double crossValidationAccuracy =
    im_training_leave_one_out(bdd,
			      trainingPeople, peopleBOW,
			      minC, maxC, // ln(min/max C)
			      minG, maxG, // ln(min/max G)
			      svmParameter);
  // N.B: when we export the model to the robot, peopleBOW index = trainingPeople
  int nrActivities = bdd.getActivities().size();
  
  // We do the training with the training people 
	struct svm_problem trainingProblem;
	trainingProblem.l = 0;
	trainingProblem.x = NULL;
	trainingProblem.y = NULL;
	for(std::vector<std::string>::const_iterator trainingPerson = trainingPeople.begin();
	    trainingPerson != trainingPeople.end();
	    ++ trainingPerson){
	  for(int i=0 ; i < peopleBOW.at(*trainingPerson).l ; i++){
	    struct svm_node* bow = peopleBOW.at(*trainingPerson).x[i];
	    addBOW(bow, peopleBOW.at(*trainingPerson).y[i],trainingProblem);
	  }
	}
	struct svm_model** svmModels = svm_train_ovr(&trainingProblem,&svmParameter);

  // Exporting models
  std::cout << "Saving the SVM model..." << std::endl;
  std::vector <std::string> modelFiles;
  for(int i=0 ; i< nrActivities ; i++){
    std::string fileToSaveModel = path2bdd;
    std::stringstream ss;
    ss << i;
    fileToSaveModel = fileToSaveModel + "/svm_ovr_" + ss.str() + ".model";
    svm_save_model(fileToSaveModel.c_str(),svmModels[i]);
    modelFiles.push_back(fileToSaveModel);
  }
  bdd.changeSVMSettings(nrActivities,	modelFiles);

  // Releasing OVR models
  for(int i=0;i<nrActivities;i++){
    svm_free_and_destroy_model(&svmModels[i]);}
  delete [] svmModels;
  svmModels = NULL;

  // Importing the models. This is for simulating the process on the robot
  struct svm_model** pSVMModels = new svm_model*[nrActivities];
  std::vector<std::string> modelFiles_import(bdd.getModelFiles());
  int j=0;
  for (std::vector<std::string>::iterator it = modelFiles_import.begin() ; it != modelFiles_import.end() ; ++it){
    std::cout << (*it) << std::endl;
    pSVMModels[j]= svm_load_model((*it).c_str());
    j++;
  }
  std::cout << "SVM models imported..." << std::endl;


  // Calculate the confusion matrix and the probability estimation
  std::cout << "Filling the training confusion matrix..." << std::endl;
  im_fill_confusion_matrix(bdd,trainingProblem,pSVMModels, trainMC);
  
  if(testingPeople.size() > 0){
    std::cout << "Filling the testing confusion matrix..." << std::endl;
    struct svm_problem testingProblem;
    testingProblem.l = 0;
    testingProblem.x = NULL;
    testingProblem.y = NULL;
    std::cout << "Testing people: ";
    for(std::vector<std::string>::const_iterator testingPerson = testingPeople.begin();
	    testingPerson != testingPeople.end();
	    ++ testingPerson){
      std::cout << *testingPerson << " ";
      for(int i=0 ; i < peopleBOW.at(*testingPerson).l ; i++){
	      struct svm_node* bow = peopleBOW.at(*testingPerson).x[i];
	      addBOW(bow, peopleBOW.at(*testingPerson).y[i],testingProblem);
      }
    }
    std::cout << std::endl;
    im_fill_confusion_matrix(bdd,testingProblem,pSVMModels, testMC);
    destroy_svm_problem(testingProblem);
  }
  
  // Releasing OVR models
  for(int i=0;i<nrActivities;i++){
    svm_free_and_destroy_model(&pSVMModels[i]);}
  delete [] pSVMModels;
  pSVMModels = NULL;


  // Releasing the svm problem, this must be done AFTER releasing the corrsponding models 
  destroy_svm_problem(trainingProblem);

  std::cout << "Releasing peopleBOW" << std::endl;
  // Releasing peopleBOW
  std::vector<std::string> people = bdd.getPeople();
  for(std::vector<std::string>::iterator person = people.begin();
      person != people.end();
      ++ person){
    destroy_svm_problem(peopleBOW[*person]);
  }

  return crossValidationAccuracy;
}

void im_compute_bdd_bow(const IMbdd& bdd, std::map <std::string, struct svm_problem>& peopleBOW){
  std::string path2bdd(bdd.getFolder());
  std::vector<std::string> activities = bdd.getActivities();
  std::vector<std::string> people = bdd.getPeople();
  std::vector<std::string> testpeople = bdd.getTestPeople();
  for(std::vector<std::string>::iterator person = testpeople.begin();
      person != testpeople.end();
      ++person){
    people.push_back(*person);  
  }

  int dim = bdd.getDim();
  int maxPts = bdd.getMaxPts();
  int k = bdd.getK();
  
  for(std::vector<std::string>::iterator person = people.begin();
      person != people.end();
      ++person){
    int currentActivity = 1;
    struct svm_problem svmPeopleBOW;
    svmPeopleBOW.l = 0;
    svmPeopleBOW.x = NULL;
    svmPeopleBOW.y = NULL;
    std::cout << "Computing the svmProblem of " << *person << std::endl;
    for(std::vector<std::string>::iterator activity = activities.begin();
	      activity != activities.end();
	      ++activity){
      string rep(path2bdd + "/" + *person + "/" + *activity + "/fp");
      DIR * repertoire = opendir(rep.c_str());
      if (!repertoire){
	      std::cerr << "Impossible to open the feature points directory!" << std::endl;
	      exit(EXIT_FAILURE);
      }
      struct dirent * ent;
      while ( (ent = readdir(repertoire)) != NULL){
	      std::string file = ent->d_name;
	      if(file.compare(".") != 0 && file.compare("..") != 0){
	        std::string path2FPs(rep + "/" + file);
	        KMdata dataPts(dim,maxPts);
	        int nPts = importSTIPs(path2FPs, dim, maxPts, &dataPts);
	        if(nPts != 0){
	          dataPts.setNPts(nPts);
	          dataPts.buildKcTree();
	          
	          KMfilterCenters ctrs(k, dataPts);
	          importCenters(path2bdd + "/" + "training.means", dim, k, &ctrs);
	          
	          // Only one BOW
	          struct svm_problem svmBow = computeBOW(currentActivity,
	      					   dataPts,
	      					   ctrs);
	          addBOW(svmBow.x[0], svmBow.y[0], svmPeopleBOW);
	          destroy_svm_problem(svmBow);	  
	        }
	      }
      }
      closedir(repertoire);
      currentActivity++;
    }
    peopleBOW.insert(std::make_pair<std::string, struct svm_problem>((*person), svmPeopleBOW));
  }
}

void im_normalize_bdd_bow(const IMbdd& bdd, const std::vector<std::string>& people,
			  std::map<std::string, struct svm_problem>& peopleBOW){
  std::string normalization(bdd.getNormalization());
  std::cout << "Doing the " << normalization << " normalization..." << std::endl;
  for(std::vector<std::string>::const_iterator person = people.begin();
	    person != people.end();
	    ++person){
    bow_normalization(bdd,peopleBOW[*person]);
  }
}

void im_compute_gaussian_param(const IMbdd& bdd, const std::vector<std::string>& trainingPeople,
			  std::map<std::string, struct svm_problem>& peopleBOW){
  int k = bdd.getK();
  struct svm_problem svmProblem;
  svmProblem.l = 0;
  svmProblem.x = NULL;
  svmProblem.y = NULL;
  // Put the correponding BOWs in the svmProblem
  for(std::vector<std::string>::const_iterator person = trainingPeople.begin();
      person != trainingPeople.end();
      ++person){
    for(int i=0 ; i<peopleBOW[*person].l ; i++){
      svm_node *bow = peopleBOW[*person].x[i];
      addBOW(bow, peopleBOW[*person].y[i],  svmProblem);
    }
  }
  double* means = new double[k];
  double* stand_devia = new double[k];
  std::cout << "Extracting gaussian parameters..." << std::endl;
  get_gaussian_parameters(k,
			  svmProblem,
			  means,
			  stand_devia);
  std::cout<<"Exporting gaussian parameters..." << std::endl;
  save_gaussian_parameters(bdd,
			   means,
			   stand_devia);
  delete []means;
  delete []stand_devia;
  destroy_svm_problem(svmProblem);
}

void bow_normalization(const IMbdd& bdd, struct svm_problem& svmProblem){
  std::string normalization(bdd.getNormalization());
  if(normalization.compare("simple") == 0 || normalization.compare("both") == 0)
    bow_simple_normalization(svmProblem);
  if(normalization.compare("gaussian") == 0|| normalization.compare("both") == 0){
    int k = bdd.getK();
    double means[k], stand_devia[k];
    load_gaussian_parameters(bdd,
			     means,
			     stand_devia);
    bow_gaussian_normalization(k,
			       means,
			       stand_devia,
			       svmProblem);
  }
}

void im_fill_confusion_matrix(const IMbdd& bdd,
			      const struct svm_problem& svmProblem,
			      struct svm_model** svmModels,
			      MatrixC& MC){
  int nrActivities = bdd.getActivities().size();
  double* py = svmProblem.y;
  int pnum = svmProblem.l;
  struct svm_node** px = svmProblem.x;
  for(int i=0 ; i<pnum ; i++){
    double* probs = new double[nrActivities];
    double lab_in = py[i];
    double lab_out = svm_predict_ovr_probs(svmModels,px[i],nrActivities,
					   probs,2);
    MC.addTransfer(lab_in,lab_out);
    delete [] probs;
  }
  std::cout << std::endl;
}


double chi_square_dist(const svm_node *px, const svm_node *py){
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += (px->value - py->value)*(px->value - py->value) / (px->value + py->value);
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index){
        sum += py->value;
				++py;
      }
			else{
        sum += px->value;
				++px;
      }
		}			
	}
  sum /= 2;
  return sum;
}


void compute_rbfchis_A(const std::vector<std::string> &trainingPeople, 
                         struct svm_parameter &svmParameter,
                         const std::map <std::string, struct svm_problem>& peopleBOW){
 	struct svm_problem trainingProblem;
	trainingProblem.l = 0;
	trainingProblem.x = NULL;
	trainingProblem.y = NULL;
	for(std::vector<std::string>::const_iterator trainingPerson = trainingPeople.begin();
	    trainingPerson != trainingPeople.end();
	    ++ trainingPerson){
	  for(int i=0 ; i < peopleBOW.at(*trainingPerson).l ; i++){
	    struct svm_node* bow = peopleBOW.at(*trainingPerson).x[i];
	    addBOW(bow, peopleBOW.at(*trainingPerson).y[i],trainingProblem);
	  }
	}
  double A = 0;
  int count = 0;
  for(int i=0 ; i<trainingProblem.l ; i++){
    for(int j=i+1 ; j<trainingProblem.l ; j++){
      count++;
      A += chi_square_dist(trainingProblem.x[i],trainingProblem.x[j]);
    }
  }
  svmParameter.rbfchisA = A/count;
	destroy_svm_problem(trainingProblem);
}


