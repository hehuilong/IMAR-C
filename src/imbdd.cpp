#include "imbdd.h"

void IMbdd::write_bdd_configuration(std::string pFolder,
				    std::string pFilename){  
  TiXmlDocument doc;  
  TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );  
  doc.LinkEndChild( decl );  
  
  TiXmlElement* root = new TiXmlElement("BDD");  
  root->SetAttribute("name",(this->bddName).c_str());
  root->SetAttribute("folder",(this->folder).c_str());
  doc.LinkEndChild( root );  
  
  TiXmlComment* bdd_comment = new TiXmlComment();
  bdd_comment->SetValue("Configuration of the BDD");  
  root->LinkEndChild(bdd_comment);  
  
  // Data
  TiXmlElement* data = new TiXmlElement("Data");
  root->LinkEndChild(data);
  
  TiXmlElement* activities = new TiXmlElement("Activities");
  TiXmlElement* activity;
  data->LinkEndChild(activities);
  for (std::vector<std::string>::iterator it = this->activities.begin() ;
       it != this->activities.end();
       ++it){
    activity = new TiXmlElement("Activitiy");
    activity->SetAttribute("name",(*it).c_str());
    activities->LinkEndChild(activity);
  }
  
  TiXmlElement* people = new TiXmlElement("People");
  TiXmlElement* person;
  data->LinkEndChild(people);
  for (std::vector<std::string>::iterator it = this->people.begin() ;
       it != this->people.end();
       ++it){
    person = new TiXmlElement("Person");
    person->SetAttribute("name",(*it).c_str());
    people->LinkEndChild(person);
  }
  
  TiXmlElement* testPeople = new TiXmlElement("TestPeople");
  TiXmlElement* testperson;
  data->LinkEndChild(testPeople);
  for (std::vector<std::string>::iterator it = this->testPeople.begin() ;
       it != this->testPeople.end();
       ++it){
    testperson = new TiXmlElement("TestPerson");
    testperson->SetAttribute("name",(*it).c_str());
    testPeople->LinkEndChild(testperson);
  }

  TiXmlElement* reject = new TiXmlElement("Reject");
  reject->SetAttribute("name",this->reject.c_str());
  data->LinkEndChild(reject);
  
  // DenseTrack
  TiXmlElement* fp = new TiXmlElement("DenseTrack");  
  root->LinkEndChild(fp);  
  
  TiXmlComment* dt_comment = new TiXmlComment();
  dt_comment->SetValue("The feature points extractor");
  fp->LinkEndChild(dt_comment);
  
  TiXmlElement* scale = new TiXmlElement("Scale");
  scale->SetAttribute("scale_num",this->scale_num);
  fp->LinkEndChild(scale);
  
  TiXmlElement* descriptor = new TiXmlElement("Descriptor");
  descriptor->SetAttribute("type",(this->descriptor).c_str());
  descriptor->SetAttribute("dim",this->dim);
  // or mbh or hoghof_mbh
  fp->LinkEndChild(descriptor);
  
  // KMeans
  TiXmlElement * kmeans = new TiXmlElement("KMeans");  
  kmeans->SetAttribute("algorithm",(this->km_algorithm).c_str());
  root->LinkEndChild(kmeans);  
  
  TiXmlElement* centers = new TiXmlElement("Centers");  
  kmeans->LinkEndChild(centers);  
  
  TiXmlElement* k = new TiXmlElement("k");
  k->SetAttribute("nr",this->k);
  centers->LinkEndChild(k);
  
  TiXmlElement* kFile = new TiXmlElement("file");
  kFile->SetAttribute("path",(this->KMeansFile).c_str());  
  centers->LinkEndChild(kFile);

  // Normalization
  TiXmlElement* normalization = new TiXmlElement("Normalization");
  normalization->SetAttribute("type",(this->normalization).c_str());
  root->LinkEndChild(normalization);
  
  TiXmlElement* means = new TiXmlElement("Means");  
  means->SetAttribute("path",(this->meansFile).c_str());  
  normalization->LinkEndChild(means);  

  TiXmlElement* standardDeviation = new TiXmlElement("StandardDeviation");
  standardDeviation->SetAttribute("path",(this->standardDeviationFile).c_str());
  normalization->LinkEndChild(standardDeviation);

  // SVM
  TiXmlElement* svm = new TiXmlElement("SVM");  
  root->LinkEndChild(svm);  
  
  TiXmlElement* nrClass = new TiXmlElement("Class");
  nrClass->SetAttribute("nr",this->nr_class);
  svm->LinkEndChild(nrClass);
  
  TiXmlElement* models = new TiXmlElement("Models");
  TiXmlElement* model;
  svm->LinkEndChild(models);
  for (std::vector<std::string>::iterator it = this->modelFiles.begin() ; it != modelFiles.end(); ++it){
    model = new TiXmlElement("Model");
    model->SetAttribute("path",(*it).c_str());
    models->LinkEndChild(model);
  }
  
  //dump_to_stdout( &doc );
  std::string savePath(pFolder + "/" + pFilename);
  doc.SaveFile(savePath.c_str());  
} 

void IMbdd::load_bdd_configuration(std::string pFolder, std::string pFilename){
  TiXmlDocument doc("imconfig");
  std::string loadPath(pFolder + "/" + pFilename);
  if(!doc.LoadFile(loadPath.c_str())){
    std::cerr << "Impossible to load the bdd!" << std::endl;
    return;
  }
  
  TiXmlHandle hDoc(&doc);
  TiXmlElement* pParent;
  TiXmlElement* pElem;
  TiXmlHandle hRoot(0);
  
  // BDD name
  pElem=hDoc.FirstChildElement().Element();
  // should always have a valid root but handle gracefully if it does
  if (!pElem) return;
  this->bddName = pElem->Attribute("name");
  this->folder = pElem->Attribute("folder");
  hRoot=TiXmlHandle(pElem);

  std::cout << "Bdd name: " << this->bddName << " Folder: " << this->folder << std::endl;
  
  // Data Setting
  pElem = hRoot.FirstChild("Data").FirstChild("Activities").FirstChild().Element();
  for(pElem; pElem; pElem=pElem->NextSiblingElement())
    this->activities.push_back(pElem->Attribute("name"));
  this->nrActivities = activities.size();
  pElem = hRoot.FirstChild("Data").FirstChild("People").FirstChild().Element();
  std::cout << "Training People: ";
  for(pElem; pElem; pElem=pElem->NextSiblingElement()){
    this->people.push_back(pElem->Attribute("name"));
    std::cout << pElem->Attribute("name") << " ";
  }
  std::cout << std::endl << "Test People: ";
  pElem = hRoot.FirstChild("Data").FirstChild("TestPeople").FirstChild().Element();
  for(pElem; pElem; pElem=pElem->NextSiblingElement()){
    this->testPeople.push_back(pElem->Attribute("name"));
    std::cout << pElem->Attribute("name") << " ";
  }
  std::cout << std::endl;
  pElem = hRoot.FirstChild("Data").FirstChild("Reject").Element();
  this->reject = pElem->Attribute("name");
  
  // DenseTrack
  pElem = hRoot.FirstChild("DenseTrack").FirstChild("Scale").Element();
  pElem->QueryIntAttribute("scale_num",&this->scale_num);
  pElem = hRoot.FirstChild("DenseTrack").FirstChild("Descriptor").Element();
  this->descriptor = pElem->Attribute("type");
  pElem->QueryIntAttribute("dim",&this->dim);
  
  // KMeans
  pElem = hRoot.FirstChildElement("KMeans").Element();
  this->km_algorithm = pElem->Attribute("algorithm");
  pElem = hRoot.FirstChild("KMeans").FirstChild().FirstChild().Element(); 
  pElem->QueryIntAttribute("nr", &k);  
  pElem = pElem->NextSiblingElement();
  this->KMeansFile = pElem->Attribute("path");
  
  // Normalization
  pElem = hRoot.FirstChild("Normalization").Element();
  this->normalization = pElem->Attribute("type");
  pElem = hRoot.FirstChild("Normalization").FirstChild("Means").Element(); 
  this->meansFile = pElem->Attribute("path");
  pElem = hRoot.FirstChild("Normalization").FirstChild("StandardDeviation").Element(); 
  this->standardDeviationFile = pElem->Attribute("path");
  
  // SVM
  pElem = hRoot.FirstChild("SVM").FirstChild("Class").Element(); 
  pElem->QueryIntAttribute("nr",&this->nr_class);
  pElem = hRoot.FirstChild("SVM").FirstChild("Models").FirstChild().Element(); 
  for(pElem; pElem; pElem=pElem->NextSiblingElement())
    this->modelFiles.push_back(pElem->Attribute("path"));
}
void IMbdd::show_bdd_configuration(){
  std::cout << "BDD: " << bddName << " (in "<< folder << ")" << std::endl;
  std::cout << "# DenseTrack" << std::endl;
  std::cout << "\t - Number of scales: " << scale_num << std::endl;
  std::cout << "\t - Descriptor: " << descriptor << std::endl;
  std::cout << "\t - Dimension: " << dim << std::endl;
  std::cout << "# KMeans" << std::endl;
  std::cout << "\t - MaxPts= " << maxPts << std::endl;
  std::cout << "\t - Algorithm: " << km_algorithm << std::endl;
  std::cout << "\t - Number of means: " << k << std::endl;
  std::cout << "\t - File to the means: " << KMeansFile << std::endl;
  std::cout << "# Normalization" << std::endl;
  std::cout << "\t - Normalization used: " << normalization << std::endl;
  std::cout << "# SVM" << std::endl;
}
void IMbdd::saveName(std::string bddName){this->bddName = bddName;};
void IMbdd::changeDataSettings(std::vector<std::string> activities,
			       std::vector<std::string> people,
			       std::vector<std::string> testPeople,
			       std::string reject){
  this->activities = activities;
  this->nrActivities = activities.size();
  this->people = people;
  this->testPeople = testPeople;
  this->reject = reject;
}
void IMbdd::changeDenseTrackSettings(int scale_num,
				     std::string descriptor,
				     int dim){
  this->scale_num = scale_num;
  this->descriptor = descriptor;
  this->dim = dim;
}
void IMbdd::changeKMSettings(std::string algorithm,
			     int k,
			     std::string KMeansFile){
  this->km_algorithm = algorithm;
  this->k = k;
  this->KMeansFile = KMeansFile;
}
void IMbdd::changeNormalizationSettings(std::string normalization,
					std::string meansFile,
					std::string standardDeviationFile){
  this->normalization = normalization;
  this->meansFile = meansFile;
  this->standardDeviationFile = standardDeviationFile;
}
void IMbdd::changeSVMSettings(int nr_class,
			      std::vector<std::string> modelFiles){
  this->nr_class = nr_class;
  this->modelFiles = modelFiles;
}
IMbdd::IMbdd(std::string bddName, std::string folder){
  this->bddName = bddName;
  this->folder = folder;
  
  // Data
  this->reject = reject;

  // DenseTrack
  this->scale_num = -1;
  this->descriptor = "";
  this->dim = -1;
  
  // KMeans
  this->maxPts = 200000;
  this->km_algorithm = "";
  this->k = -1;
  this->KMeansFile = "";
  
  // Normalization
  this->normalization = "";
  this->meansFile = "";
  this->standardDeviationFile = "";
  
  // SVM
  this->nr_class = -1;
}
