#include "tactil.h"
#include <time.h>

int nbOfFiles(std::string path){
  int nbFiles = 0; 
  DIR * repertoire = opendir(path.c_str());
  if(!repertoire) std::cerr << "Impossible to open the directory!" << std::endl;
  else{
    struct dirent * ent;
    while ( (ent = readdir(repertoire)) != NULL){
      std::string file = ent->d_name;
      if(file.compare(".") != 0 && file.compare("..") != 0) nbFiles++;
    }
    closedir(repertoire);
  }
  return nbFiles;
}

Tactil::Tactil(boost::shared_ptr<AL::ALBroker> broker,
	       const std::string& name):
  AL::ALModule(broker, name),fCallbackMutex(AL::ALMutex::createALMutex()){
  
  // A little description:
  setModuleDescription("This module presents how to subscribe to a simple event (here FrontTactilTouched) and use a callback method.");
  
  // Method name
  functionName("onFrontHeadTouched", // function name
               getName(), // module name
               "Method when the robot front is touched."); // description
  
  BIND_METHOD(Tactil::onFrontHeadTouched); // to made the method appeared in the web interface
}

Tactil::~Tactil() {
  fMemoryProxy.unsubscribeToEvent("FrontTactilTouched", // event name
                                  "Tactil"); // module name
  fMemoryProxy.unsubscribeToEvent("startRecognition", // event name
                                  "Tactil"); // module name
}

void Tactil::init() {
  try {
    // Create a proxy to ALMemory.
    fMemoryProxy = AL::ALMemoryProxy(getParentBroker());
    
    // subscribing to event
    fMemoryProxy.subscribeToEvent("FrontTactilTouched",     // event name
                                  "Tactil",                 // module name
                                  "onFrontHeadTouched");  //method name
    fMemoryProxy.subscribeToEvent("startRecognition",
                                  "Tactil",
                                  "onFrontHeadTouched");
  }
  catch (const AL::ALError& e) {
    qiLogError("module.example") << e.what() << std::endl;
  }
  
  rpp = AL::ALRobotPostureProxy(getParentBroker());
  rpp.goToPosture("Crouch",0.5f);
  
  videoRecorderProxy = AL::ALVideoRecorderProxy(getParentBroker());

  ttsp = AL::ALTextToSpeechProxy(getParentBroker());
  ttsp.setLanguage("French");
  ttsp.setVoice( "Kenny22Enhanced");
}

void Tactil::onFrontHeadTouched(){
    // unsubscribing to event to avoid loops
    fMemoryProxy.unsubscribeToEvent("FrontTactilTouched", // event name
                                    "Tactil"); // module name
    fMemoryProxy.unsubscribeToEvent("startRecognition", // event name
                                    "Tactil"); // module name
    qiLogInfo("Tactil") << "Executing callback method on FrontTactilTouched event" << std::endl;

    // As long as this is defined, the code is thread-safe
    AL::ALCriticalSection section(fCallbackMutex);

    // Check that the front is touched
  if((float) fMemoryProxy.getData("FrontTactilTouched") == 1.0f){
    fState = fMemoryProxy.getData("FrontTactilTouched");
  }
  else{
    fState =  fMemoryProxy.getData("startRecognition");
  }
 
  if (fState) {
      lp = AL::ALLedsProxy(getParentBroker());

      // post permits to execute a long method and to continue the programm
      // the id returned permits to  stop the methode with the stop function
      // int idLeds = lp.post.rotateEyes(0x0000CC00,0.5,100.0); // long time in the second
      int idLeds = lp.post.rasta(100.0); // long time in the second

      // The duration of the video captured
      int duration = 2;
      videoRecorderProxy.setResolution(1);
      // kQQVGA ( 160 * 120 ) 0
      // kQVGA (320 * 240) 1
      // kVGA (640 * 480) 2
      videoRecorderProxy.setFrameRate(12);

      //std::string videoOutput("recording-");
      int nrVideos = nbOfFiles("/home/nao/recordings/cameras/TEST");
      std::string videoOutput = inttostring(nrVideos);
      // Warning: it can delete existant videos

      sleep(duration);
      ttsp.post.say("Action");
      // Recording video
      videoRecorderProxy.startRecording("/home/nao/recordings/cameras/TEST",videoOutput);
      sleep(duration);
      AL::ALValue video = videoRecorderProxy.stopRecording();
      ttsp.post.say("End recording");

      // Decreasing quality into kQQVGA for the treatment
      videoOutput = videoOutput + ".f12s160.avi";
      decrease_quality(video[1],"/home/nao/recordings/cameras/TEST/" + videoOutput);

      long starttime, finishtime;
      double totaltime;
      qiLogInfo("Tactil") << "Predicting activity..." << std::endl;
      starttime = clock();
      std::string activityPredicted = integration("/home/nao/recordings/cameras/TEST/" + videoOutput,
                                                  "/home/nao/data/activity_recognition");
      finishtime = clock();
      totaltime = finishtime - starttime;
      totaltime /= CLOCKS_PER_SEC;
      std::cerr<<"["<<totaltime<<" seconds]"<<std::endl;
      lp.stop(idLeds);
      lp.on("FaceLeds");
      //lp.off("LeftFaceLedsGreen");
      //lp.off("RightFaceLedsGreen");
      if(activityPredicted.compare("applaud") == 0){
         applaudAnimation();
         rpp.goToPosture("Crouch",0.5f);
     }
     qiLogInfo("Tactil") << "Activitée  détectée : " << activityPredicted << std::endl;
     ttsp.post.say("Activity detected: " + activityPredicted);
     lp.on("FaceLeds");
 }



  // subscribing again to events
  fMemoryProxy.subscribeToEvent("FrontTactilTouched",     // event name
                                "Tactil",                 // module name
                                "onFrontHeadTouched");  //method name
  fMemoryProxy.subscribeToEvent("startRecognition",
                                "Tactil",
                                "onFrontHeadTouched");
}
void Tactil::decrease_quality(std::string vInput, std::string vOutput){
    cv::VideoCapture capture(vInput);
    double fpsOut = 12;
    double fpsIn = capture.get(CV_CAP_PROP_FPS);

    cv::Size fsize(160,120);
    // 120*106 = QQVGA Quarter-QVGA
    // 320*240 = QVGA Quarter Video Graphics Array
    // 640*480 = VGA Video Graphics Array

    int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fourcc = capture.get(CV_CAP_PROP_FOURCC);

    cv::VideoWriter writer(vOutput,
                           fourcc,
                           fpsOut,
                           fsize);
    if(!writer.isOpened()){
      std::cerr << "Error while opening Writer!" << std::endl;
    }
    cv::Mat mat;
    cv::Mat mConverted(fsize.height,fsize.width, CV_8UC1);
    int maxFrames = frameCount;

    // Randomizing the frames
    int values[100];
    for(int i=0 ; i<frameCount ; i++){
      values[i] = i;
    }

    for(int i=0 ; i<maxFrames ; i++){
      capture.set(CV_CAP_PROP_POS_FRAMES, (double) values[i]);
      capture >> mat;
      resize(mat,mConverted,fsize);
      writer << mConverted;
    }

    std::cout << "INPUT: " << vInput << std::endl;
    std::cout << "* Number of frames: " << frameCount << std::endl;
    std::cout << "* Frames per second: " << fpsIn << std::endl;

    std::cout << "OUTPUT: " << vOutput << std::endl;
    std::cout << "* Number of frames: " << maxFrames << std::endl;
    std::cout << "* Frames per second: " << fpsOut << std::endl;
}

void Tactil::helloAnimation(){
// Saving previous posture
//AL::ALRobotPostureProxy previousPosture(getParentBroker());

// Choregraphe bezier export in c++.
std::vector<std::string> names;
AL::ALValue times, keys;
names.reserve(14);
times.arraySetSize(14);
keys.arraySetSize(14);

names.push_back("HeadPitch");
times[0].arraySetSize(6);
keys[0].arraySetSize(6);

times[0][0] = 0.800000;
keys[0][0] = AL::ALValue::array(0.296020, AL::ALValue::array(3, -0.266667, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[0][1] = 1.56000;
keys[0][1] = AL::ALValue::array(-0.170316, AL::ALValue::array(3, -0.253333, 0.111996), AL::ALValue::array(3, 0.226667, -0.100207));
times[0][2] = 2.24000;
keys[0][2] = AL::ALValue::array(-0.340591, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[0][3] = 2.80000;
keys[0][3] = AL::ALValue::array(-0.0598679, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[0][4] = 3.48000;
keys[0][4] = AL::ALValue::array(-0.193327, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[0][5] = 4.60000;
keys[0][5] = AL::ALValue::array(-0.0107800, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("HeadYaw");
times[1].arraySetSize(6);
keys[1].arraySetSize(6);

times[1][0] = 0.800000;
keys[1][0] = AL::ALValue::array(-0.135034, AL::ALValue::array(3, -0.266667, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[1][1] = 1.56000;
keys[1][1] = AL::ALValue::array(-0.351328, AL::ALValue::array(3, -0.253333, 0.0493864), AL::ALValue::array(3, 0.226667, -0.0441878));
times[1][2] = 2.24000;
keys[1][2] = AL::ALValue::array(-0.415757, AL::ALValue::array(3, -0.226667, 0.00372364), AL::ALValue::array(3, 0.186667, -0.00306653));
times[1][3] = 2.80000;
keys[1][3] = AL::ALValue::array(-0.418823, AL::ALValue::array(3, -0.186667, 0.00306653), AL::ALValue::array(3, 0.226667, -0.00372364));
times[1][4] = 3.48000;
keys[1][4] = AL::ALValue::array(-0.520068, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[1][5] = 4.60000;
keys[1][5] = AL::ALValue::array(-0.375872, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LElbowRoll");
times[2].arraySetSize(6);
keys[2].arraySetSize(6);

times[2][0] = 0.720000;
keys[2][0] = AL::ALValue::array(-1.37902, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[2][1] = 1.48000;
keys[2][1] = AL::ALValue::array(-1.29005, AL::ALValue::array(3, -0.253333, -0.0345436), AL::ALValue::array(3, 0.226667, 0.0309074));
times[2][2] = 2.16000;
keys[2][2] = AL::ALValue::array(-1.18267, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[2][3] = 2.72000;
keys[2][3] = AL::ALValue::array(-1.24863, AL::ALValue::array(3, -0.186667, 0.0205524), AL::ALValue::array(3, 0.226667, -0.0249565));
times[2][4] = 3.40000;
keys[2][4] = AL::ALValue::array(-1.31920, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[2][5] = 4.52000;
keys[2][5] = AL::ALValue::array(-1.18421, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LElbowYaw");
times[3].arraySetSize(6);
keys[3].arraySetSize(6);

times[3][0] = 0.720000;
keys[3][0] = AL::ALValue::array(-0.803859, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[3][1] = 1.48000;
keys[3][1] = AL::ALValue::array(-0.691876, AL::ALValue::array(3, -0.253333, -0.0137171), AL::ALValue::array(3, 0.226667, 0.0122732));
times[3][2] = 2.16000;
keys[3][2] = AL::ALValue::array(-0.679603, AL::ALValue::array(3, -0.226667, -0.0122732), AL::ALValue::array(3, 0.186667, 0.0101073));
times[3][3] = 2.72000;
keys[3][3] = AL::ALValue::array(-0.610574, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[3][4] = 3.40000;
keys[3][4] = AL::ALValue::array(-0.753235, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[3][5] = 4.52000;
keys[3][5] = AL::ALValue::array(-0.670400, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LHand");
times[4].arraySetSize(2);
keys[4].arraySetSize(2);

times[4][0] = 1.48000;
keys[4][0] = AL::ALValue::array(0.00415750, AL::ALValue::array(3, -0.493333, -0.00000), AL::ALValue::array(3, 1.01333, 0.00000));
times[4][1] = 4.52000;
keys[4][1] = AL::ALValue::array(0.00418923, AL::ALValue::array(3, -1.01333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LShoulderPitch");
times[5].arraySetSize(6);
keys[5].arraySetSize(6);

times[5][0] = 0.720000;
keys[5][0] = AL::ALValue::array(1.11824, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[5][1] = 1.48000;
keys[5][1] = AL::ALValue::array(0.928028, AL::ALValue::array(3, -0.253333, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[5][2] = 2.16000;
keys[5][2] = AL::ALValue::array(0.940300, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[5][3] = 2.72000;
keys[5][3] = AL::ALValue::array(0.862065, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[5][4] = 3.40000;
keys[5][4] = AL::ALValue::array(0.897349, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[5][5] = 4.52000;
keys[5][5] = AL::ALValue::array(0.842125, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LShoulderRoll");
times[6].arraySetSize(6);
keys[6].arraySetSize(6);

times[6][0] = 0.720000;
keys[6][0] = AL::ALValue::array(0.363515, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[6][1] = 1.48000;
keys[6][1] = AL::ALValue::array(0.226991, AL::ALValue::array(3, -0.253333, 0.0257175), AL::ALValue::array(3, 0.226667, -0.0230104));
times[6][2] = 2.16000;
keys[6][2] = AL::ALValue::array(0.203980, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[6][3] = 2.72000;
keys[6][3] = AL::ALValue::array(0.217786, AL::ALValue::array(3, -0.186667, -0.00669692), AL::ALValue::array(3, 0.226667, 0.00813198));
times[6][4] = 3.40000;
keys[6][4] = AL::ALValue::array(0.248467, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[6][5] = 4.52000;
keys[6][5] = AL::ALValue::array(0.226991, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("LWristYaw");
times[7].arraySetSize(2);
keys[7].arraySetSize(2);

times[7][0] = 1.48000;
keys[7][0] = AL::ALValue::array(0.147222, AL::ALValue::array(3, -0.493333, -0.00000), AL::ALValue::array(3, 1.01333, 0.00000));
times[7][1] = 4.52000;
keys[7][1] = AL::ALValue::array(0.119610, AL::ALValue::array(3, -1.01333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RElbowRoll");
times[8].arraySetSize(10);
keys[8].arraySetSize(10);

times[8][0] = 0.640000;
keys[8][0] = AL::ALValue::array(1.38524, AL::ALValue::array(3, -0.213333, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[8][1] = 1.40000;
keys[8][1] = AL::ALValue::array(0.242414, AL::ALValue::array(3, -0.253333, -0.00000), AL::ALValue::array(3, 0.0933333, 0.00000));
times[8][2] = 1.68000;
keys[8][2] = AL::ALValue::array(0.349066, AL::ALValue::array(3, -0.0933333, -0.0949577), AL::ALValue::array(3, 0.133333, 0.135654));
times[8][3] = 2.08000;
keys[8][3] = AL::ALValue::array(0.934249, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.106667, 0.00000));
times[8][4] = 2.40000;
keys[8][4] = AL::ALValue::array(0.680678, AL::ALValue::array(3, -0.106667, 0.141383), AL::ALValue::array(3, 0.0800000, -0.106037));
times[8][5] = 2.64000;
keys[8][5] = AL::ALValue::array(0.191986, AL::ALValue::array(3, -0.0800000, -0.00000), AL::ALValue::array(3, 0.133333, 0.00000));
times[8][6] = 3.04000;
keys[8][6] = AL::ALValue::array(0.261799, AL::ALValue::array(3, -0.133333, -0.0698132), AL::ALValue::array(3, 0.0933333, 0.0488692));
times[8][7] = 3.32000;
keys[8][7] = AL::ALValue::array(0.707216, AL::ALValue::array(3, -0.0933333, -0.103967), AL::ALValue::array(3, 0.133333, 0.148524));
times[8][8] = 3.72000;
keys[8][8] = AL::ALValue::array(1.01927, AL::ALValue::array(3, -0.133333, -0.0664734), AL::ALValue::array(3, 0.240000, 0.119652));
times[8][9] = 4.44000;
keys[8][9] = AL::ALValue::array(1.26559, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RElbowYaw");
times[9].arraySetSize(7);
keys[9].arraySetSize(7);

times[9][0] = 0.640000;
keys[9][0] = AL::ALValue::array(-0.312978, AL::ALValue::array(3, -0.213333, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[9][1] = 1.40000;
keys[9][1] = AL::ALValue::array(0.564471, AL::ALValue::array(3, -0.253333, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[9][2] = 2.08000;
keys[9][2] = AL::ALValue::array(0.391128, AL::ALValue::array(3, -0.226667, 0.0395378), AL::ALValue::array(3, 0.186667, -0.0325606));
times[9][3] = 2.64000;
keys[9][3] = AL::ALValue::array(0.348176, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[9][4] = 3.32000;
keys[9][4] = AL::ALValue::array(0.381923, AL::ALValue::array(3, -0.226667, -0.0337477), AL::ALValue::array(3, 0.133333, 0.0198516));
times[9][5] = 3.72000;
keys[9][5] = AL::ALValue::array(0.977384, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.240000, 0.00000));
times[9][6] = 4.44000;
keys[9][6] = AL::ALValue::array(0.826783, AL::ALValue::array(3, -0.240000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RHand");
times[10].arraySetSize(3);
keys[10].arraySetSize(3);

times[10][0] = 1.40000;
keys[10][0] = AL::ALValue::array(0.0148960, AL::ALValue::array(3, -0.466667, -0.00000), AL::ALValue::array(3, 0.640000, 0.00000));
times[10][1] = 3.32000;
keys[10][1] = AL::ALValue::array(0.0149214, AL::ALValue::array(3, -0.640000, -0.00000), AL::ALValue::array(3, 0.373333, 0.00000));
times[10][2] = 4.44000;
keys[10][2] = AL::ALValue::array(0.00741967, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RShoulderPitch");
times[11].arraySetSize(6);
keys[11].arraySetSize(6);

times[11][0] = 0.640000;
keys[11][0] = AL::ALValue::array(0.247016, AL::ALValue::array(3, -0.213333, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[11][1] = 1.40000;
keys[11][1] = AL::ALValue::array(-1.17193, AL::ALValue::array(3, -0.253333, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[11][2] = 2.08000;
keys[11][2] = AL::ALValue::array(-1.08910, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[11][3] = 2.64000;
keys[11][3] = AL::ALValue::array(-1.26091, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[11][4] = 3.32000;
keys[11][4] = AL::ALValue::array(-1.14892, AL::ALValue::array(3, -0.226667, -0.111982), AL::ALValue::array(3, 0.373333, 0.184441));
times[11][5] = 4.44000;
keys[11][5] = AL::ALValue::array(1.02015, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RShoulderRoll");
times[12].arraySetSize(6);
keys[12].arraySetSize(6);

times[12][0] = 0.640000;
keys[12][0] = AL::ALValue::array(-0.242414, AL::ALValue::array(3, -0.213333, -0.00000), AL::ALValue::array(3, 0.253333, 0.00000));
times[12][1] = 1.40000;
keys[12][1] = AL::ALValue::array(-0.954191, AL::ALValue::array(3, -0.253333, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[12][2] = 2.08000;
keys[12][2] = AL::ALValue::array(-0.460242, AL::ALValue::array(3, -0.226667, -0.00000), AL::ALValue::array(3, 0.186667, 0.00000));
times[12][3] = 2.64000;
keys[12][3] = AL::ALValue::array(-0.960325, AL::ALValue::array(3, -0.186667, -0.00000), AL::ALValue::array(3, 0.226667, 0.00000));
times[12][4] = 3.32000;
keys[12][4] = AL::ALValue::array(-0.328317, AL::ALValue::array(3, -0.226667, -0.0474984), AL::ALValue::array(3, 0.373333, 0.0782326));
times[12][5] = 4.44000;
keys[12][5] = AL::ALValue::array(-0.250085, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

names.push_back("RWristYaw");
times[13].arraySetSize(3);
keys[13].arraySetSize(3);

times[13][0] = 1.40000;
keys[13][0] = AL::ALValue::array(-0.312978, AL::ALValue::array(3, -0.466667, -0.00000), AL::ALValue::array(3, 0.640000, 0.00000));
times[13][1] = 3.32000;
keys[13][1] = AL::ALValue::array(-0.303775, AL::ALValue::array(3, -0.640000, -0.00920312), AL::ALValue::array(3, 0.373333, 0.00536849));
times[13][2] = 4.44000;
keys[13][2] = AL::ALValue::array(0.182504, AL::ALValue::array(3, -0.373333, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

try
{
    getParentBroker()->getMotionProxy()->angleInterpolationBezier(names, times, keys);
}
catch(const std::exception&)
{

}
}
void Tactil::applaudAnimation()
{
    // Choregraphe bezier export in c++.
    // Add #include <alproxies/almotionproxy.h> at the beginning of this file.
    std::vector<std::string> names;
    AL::ALValue times, keys;
    names.reserve(12);
    times.arraySetSize(12);
    keys.arraySetSize(12);

    names.push_back("LElbowRoll");
    times[0].arraySetSize(6);
    keys[0].arraySetSize(6);

    times[0][0] = 0.400000;
    keys[0][0] = AL::ALValue::array(-1.21182, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[0][1] = 1.00000;
    keys[0][1] = AL::ALValue::array(-1.19648, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[0][2] = 1.60000;
    keys[0][2] = AL::ALValue::array(-1.21182, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[0][3] = 2.20000;
    keys[0][3] = AL::ALValue::array(-1.19648, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[0][4] = 2.80000;
    keys[0][4] = AL::ALValue::array(-1.21182, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[0][5] = 3.40000;
    keys[0][5] = AL::ALValue::array(-1.19648, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("LElbowYaw");
    times[1].arraySetSize(6);
    keys[1].arraySetSize(6);

    times[1][0] = 0.400000;
    keys[1][0] = AL::ALValue::array(-1.63989, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[1][1] = 1.00000;
    keys[1][1] = AL::ALValue::array(-1.13827, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[1][2] = 1.60000;
    keys[1][2] = AL::ALValue::array(-1.63989, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[1][3] = 2.20000;
    keys[1][3] = AL::ALValue::array(-1.13827, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[1][4] = 2.80000;
    keys[1][4] = AL::ALValue::array(-1.63989, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[1][5] = 3.40000;
    keys[1][5] = AL::ALValue::array(-1.13827, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("LHand");
    times[2].arraySetSize(6);
    keys[2].arraySetSize(6);

    times[2][0] = 0.400000;
    keys[2][0] = AL::ALValue::array(0.00804946, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[2][1] = 1.00000;
    keys[2][1] = AL::ALValue::array(0.00807040, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[2][2] = 1.60000;
    keys[2][2] = AL::ALValue::array(0.00804946, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[2][3] = 2.20000;
    keys[2][3] = AL::ALValue::array(0.00807040, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[2][4] = 2.80000;
    keys[2][4] = AL::ALValue::array(0.00804946, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[2][5] = 3.40000;
    keys[2][5] = AL::ALValue::array(0.00807040, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("LShoulderPitch");
    times[3].arraySetSize(6);
    keys[3].arraySetSize(6);

    times[3][0] = 0.400000;
    keys[3][0] = AL::ALValue::array(0.805308, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[3][1] = 1.00000;
    keys[3][1] = AL::ALValue::array(0.711734, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[3][2] = 1.60000;
    keys[3][2] = AL::ALValue::array(0.805309, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[3][3] = 2.20000;
    keys[3][3] = AL::ALValue::array(0.711735, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[3][4] = 2.80000;
    keys[3][4] = AL::ALValue::array(0.805309, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[3][5] = 3.40000;
    keys[3][5] = AL::ALValue::array(0.711735, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("LShoulderRoll");
    times[4].arraySetSize(6);
    keys[4].arraySetSize(6);

    times[4][0] = 0.400000;
    keys[4][0] = AL::ALValue::array(0.291418, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[4][1] = 1.00000;
    keys[4][1] = AL::ALValue::array(-0.240880, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[4][2] = 1.60000;
    keys[4][2] = AL::ALValue::array(0.291418, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[4][3] = 2.20000;
    keys[4][3] = AL::ALValue::array(-0.240880, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[4][4] = 2.80000;
    keys[4][4] = AL::ALValue::array(0.291418, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[4][5] = 3.40000;
    keys[4][5] = AL::ALValue::array(-0.240880, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("LWristYaw");
    times[5].arraySetSize(6);
    keys[5].arraySetSize(6);

    times[5][0] = 0.400000;
    keys[5][0] = AL::ALValue::array(0.119610, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[5][1] = 1.00000;
    keys[5][1] = AL::ALValue::array(0.0674541, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[5][2] = 1.60000;
    keys[5][2] = AL::ALValue::array(0.119610, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[5][3] = 2.20000;
    keys[5][3] = AL::ALValue::array(0.0674542, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[5][4] = 2.80000;
    keys[5][4] = AL::ALValue::array(0.119610, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[5][5] = 3.40000;
    keys[5][5] = AL::ALValue::array(0.0674542, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RElbowRoll");
    times[6].arraySetSize(6);
    keys[6].arraySetSize(6);

    times[6][0] = 0.400000;
    keys[6][0] = AL::ALValue::array(1.05390, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[6][1] = 1.00000;
    keys[6][1] = AL::ALValue::array(1.08305, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[6][2] = 1.60000;
    keys[6][2] = AL::ALValue::array(1.05390, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[6][3] = 2.20000;
    keys[6][3] = AL::ALValue::array(1.08305, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[6][4] = 2.80000;
    keys[6][4] = AL::ALValue::array(1.05390, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[6][5] = 3.40000;
    keys[6][5] = AL::ALValue::array(1.08305, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RElbowYaw");
    times[7].arraySetSize(6);
    keys[7].arraySetSize(6);

    times[7][0] = 0.400000;
    keys[7][0] = AL::ALValue::array(1.43732, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[7][1] = 1.00000;
    keys[7][1] = AL::ALValue::array(1.02160, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[7][2] = 1.60000;
    keys[7][2] = AL::ALValue::array(1.43732, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[7][3] = 2.20000;
    keys[7][3] = AL::ALValue::array(1.02160, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[7][4] = 2.80000;
    keys[7][4] = AL::ALValue::array(1.43732, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[7][5] = 3.40000;
    keys[7][5] = AL::ALValue::array(1.02160, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RHand");
    times[8].arraySetSize(6);
    keys[8].arraySetSize(6);

    times[8][0] = 0.400000;
    keys[8][0] = AL::ALValue::array(0.00888023, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[8][1] = 1.00000;
    keys[8][1] = AL::ALValue::array(0.00886627, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[8][2] = 1.60000;
    keys[8][2] = AL::ALValue::array(0.00888024, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[8][3] = 2.20000;
    keys[8][3] = AL::ALValue::array(0.00886627, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[8][4] = 2.80000;
    keys[8][4] = AL::ALValue::array(0.00888024, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[8][5] = 3.40000;
    keys[8][5] = AL::ALValue::array(0.00886627, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RShoulderPitch");
    times[9].arraySetSize(6);
    keys[9].arraySetSize(6);

    times[9][0] = 0.400000;
    keys[9][0] = AL::ALValue::array(0.733294, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[9][1] = 1.00000;
    keys[9][1] = AL::ALValue::array(0.625914, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[9][2] = 1.60000;
    keys[9][2] = AL::ALValue::array(0.733295, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[9][3] = 2.20000;
    keys[9][3] = AL::ALValue::array(0.625914, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[9][4] = 2.80000;
    keys[9][4] = AL::ALValue::array(0.733295, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[9][5] = 3.40000;
    keys[9][5] = AL::ALValue::array(0.625914, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RShoulderRoll");
    times[10].arraySetSize(6);
    keys[10].arraySetSize(6);

    times[10][0] = 0.400000;
    keys[10][0] = AL::ALValue::array(-0.366668, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[10][1] = 1.00000;
    keys[10][1] = AL::ALValue::array(0.288350, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[10][2] = 1.60000;
    keys[10][2] = AL::ALValue::array(-0.366667, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[10][3] = 2.20000;
    keys[10][3] = AL::ALValue::array(0.288349, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[10][4] = 2.80000;
    keys[10][4] = AL::ALValue::array(-0.366667, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[10][5] = 3.40000;
    keys[10][5] = AL::ALValue::array(0.288349, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    names.push_back("RWristYaw");
    times[11].arraySetSize(6);
    keys[11].arraySetSize(6);

    times[11][0] = 0.400000;
    keys[11][0] = AL::ALValue::array(0.0797260, AL::ALValue::array(3, -0.133333, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[11][1] = 1.00000;
    keys[11][1] = AL::ALValue::array(0.0781920, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[11][2] = 1.60000;
    keys[11][2] = AL::ALValue::array(0.0797259, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[11][3] = 2.20000;
    keys[11][3] = AL::ALValue::array(0.0781920, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[11][4] = 2.80000;
    keys[11][4] = AL::ALValue::array(0.0797259, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.200000, 0.00000));
    times[11][5] = 3.40000;
    keys[11][5] = AL::ALValue::array(0.0781920, AL::ALValue::array(3, -0.200000, -0.00000), AL::ALValue::array(3, 0.00000, 0.00000));

    try
    {
      getParentBroker()->getMotionProxy()->angleInterpolationBezier(names, times, keys);
    }
    catch(const std::exception&)
    {

    }

}
