#include "imconfig.h"

/**
 * \fn void listBdds()
 * \brief List BDDs present in the global database.
 */
void listBdds(){
  DIR * repertoire = opendir("bdd");
  if (repertoire == NULL){
    std::cerr << "Impossible to open the BDDs directory!" << std::endl;
    exit(EXIT_FAILURE);
  }
  struct dirent * ent;
  while ( (ent = readdir(repertoire)) != NULL){
    std::string file = ent->d_name;
    if(file.compare(".") != 0 && file.compare("..") != 0)
      std::cout << file << std::endl;
  }
  closedir(repertoire);
}

/**
 * \fn void listActivities(std::string bdd)
 * \brief List activities present in the specified database.
 * \param[in] bdd The name of the bdd.
 */
void listActivities(std::string bdd){
  std::string path2bdd("bdd/" + bdd);   
  activitiesMap *am;
  int nbActivities = mapActivities(path2bdd,&am);
  for(int i=0 ; i<nbActivities ; i++){
    std::cout << am[i].activity << " " << "(";
    std::cout << am[i].label << ")";
    std::cout << std::endl;
  }
  delete []am;
}

/**
 * \fn int mapActivities(std::string path2bdd, activitiesMap **am)
 * \brief Fills the object activitiesMap which contain the equivalence Label-Activity.
 * \param[in] path2bdd The path to the BDD
 * \param[in,out] am A pointer to an object activitiesMap.
 * \return The number of activities.
 */
int mapActivities(std::string path2bdd, activitiesMap **am){
  path2bdd = path2bdd + "/mapping.txt";
  std::ifstream in(path2bdd.c_str(),ios::in);
  if(!in){
    std::cerr << "Can't open file mapping.txt!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<string> mapString;
  int nbActivities = 0;
  bool endOfLine = false;
  while(!(in.eof())){
    while(!endOfLine){
      std::string tmp;
      if(!(in >> tmp)){
	endOfLine = true;
      }
      if (!endOfLine){
	mapString.push_back(tmp);
	nbActivities++;
      }
    }
  }
  in.close();
  *am = new activitiesMap[mapString.size()];
  
  for(unsigned int i=0 ; i<mapString.size() ; i++){
    std::istringstream iss(mapString[i]);
    std::string token;
    int j = 0;
    while(getline(iss, token,':')){
      if(j==0){
	(*am)[i].label = atoi(token.c_str());
      }
      if(j==1){
	(*am)[i].activity = token;
      }
      j++;
    }
  }
  
  // tri insertion
  for(int i=1 ; i<nbActivities ; i++){
    activitiesMap tmp;
    tmp.activity = (*am)[i].activity;
    tmp.label = (*am)[i].label;
    
    int j = i;
    while(j>0 && (*am)[j-1].label > tmp.label){
      (*am)[j].label = (*am)[j-1].label;
      (*am)[j].activity = (*am)[j-1].activity;
      j--;
    }
    (*am)[j].label = tmp.label;
    (*am)[j].activity = tmp.activity;
  }
  
  return nbActivities;
} 
bool labelExist(int label, activitiesMap *am, int nbActivities){
  int i = 0;
  while(i<nbActivities && am[i].label != label){
    i++;
  }
  return am[i].label == label;
}
int searchMapIndex(int label, activitiesMap *am, int nbActivities){
  int i = 0;
  while(i<nbActivities && am[i].label != label){
    i++;
  }
  if(am[i].label != label){
    std::cerr << " " << am[i].label << " " << label << " File mapping.txt corrupted!" << std::endl;
    exit(EXIT_FAILURE);
  }
  return i;
}

/**
 * \fn int getDim(int desc)
 * \brief get the dimension from the descriptor number 
 *
 * \param[in] desc Descriptor ID
 * \return Feature points dimension
 */
int getDim(std::string descriptor){
  int dimHOGHOF = 204;
  int dimMBH = 192;
  // HOG HOF
  if(descriptor.compare("hoghof") == 0)
    return dimHOGHOF;
  
  // MBH
  if(descriptor.compare("mbh") == 0)
    return dimMBH;
  
  if(descriptor.compare("all") == 0)
    return dimHOGHOF + dimMBH;
  return -1;
}

/**
 * \fn double getTrainProbability(std::string folder)
 * \brief Get the training probability.
 * \param[in] folder The folder containing the probability file.
 * \return Number of centers
 */
double getTrainProbability(std::string folder){
  std::string path2file(folder + "/train_probability.txt");
  ifstream in(path2file.c_str());
  double p;
  in >> p;
  return p;
}
/**
 * \fn void saveTrainProbability(std::string folder,int k)
 * \bref Save the training probability in the folder.
 * \param[in] folder The folder which will contain kmeans.txt.
 * \param[in] k Number of means.
 */
void saveTrainProbability(std::string folder, double p){
  std::string path2file(folder + "/train_probability.txt");
  ofstream out(path2file.c_str(), ios::out);
  out << p << std::endl;
}

/**
 * \fn void saveDescInfo(string bddName,int desc)
 * \bref save the descriptor information into the BDD
 * \param[in] bddName the BDD name
 * \param[in] desc the descriptor number
 */
void saveDescInfo(string bddName,int desc){
  std::string path2file("bdd/" + bddName + "/desc.txt");
  ofstream out(path2file.c_str(), ios::out);
  out << desc << std::endl;
}

/**
 * \fn int getDescID(string bddName)
 * \brief get the descriptor number from the BDD
 * \param[in] folder The folder containing the descriptor file.
 * \return ID of the descriptor
 */
int getDescID(std::string folder){
  std::string path2file(folder + "/desc.txt");
  ifstream in(path2file.c_str());
  if(!in){
    std::cerr << "Impossible to open the descriptor file!" << std::endl;
    exit(EXIT_FAILURE);
  }
  int desc;
  in >> desc;
  return desc;
}

/**
 * \fn int getK(std::string folder)
 * \brief Get the number of means.
 * \param[in] folder The folder containing the kmeans file.
 * \return Number of centers
 */
int getK(std::string folder){
  std::string path2file(folder + "/kmeans.txt");
  ifstream in(path2file.c_str());
  int k;
  in >> k;
  return k;
}
/**
 * \fn void saveKinfo(std::string bddName,int k)
 * \bref Save the number of cluster into the foder/
 * \param[in] folder The folder which will contain kmeans.txt.
 * \param[in] k Number of means.
 */
void saveKinfo(string folder,int k){
  std::string path2file(folder + "/kmeans.txt");
  ofstream out(path2file.c_str(), ios::out);
  out << k << std::endl;
}
