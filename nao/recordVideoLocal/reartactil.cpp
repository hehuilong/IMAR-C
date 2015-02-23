
/**
 * Copyright (c) 2011 Aldebaran Robotics
 */

#include "reartactil.h"
#include <alproxies/alvideorecorderproxy.h>
#include <alvalue/alvalue.h>
#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <qi/log.hpp>
#include <althread/alcriticalsection.h>
#include <sstream>
#include <string>

RearTactil::RearTactil(
  boost::shared_ptr<AL::ALBroker> broker,
  const std::string& name): AL::ALModule(broker, name),
    fCallbackMutex(AL::ALMutex::createALMutex())
{
  setModuleDescription("This module presents how to subscribe to a simple event (here RearTactilTouched) and use a callback method.");

  functionName("onRearTactilTouched", getName(), "Method called when the rear tactil is pressed. Makes a LED animation.");
  BIND_METHOD(RearTactil::onRearTactilTouched)
}

RearTactil::~RearTactil() {
  fMemoryProxy.unsubscribeToEvent("onRearTactilTouched", "RearTactil");
}

void RearTactil::init() {
  try {
    /** Create a proxy to ALMemory.
    */
    fMemoryProxy = AL::ALMemoryProxy(getParentBroker());

    fState = fMemoryProxy.getData("RearTactilTouched");
    /** Subscribe to event LeftRearTactilPressed
    * Arguments:
    * - name of the event
    * - name of the module to be called for the callback
    * - name of the bound method to be called on event
    */
    fMemoryProxy.subscribeToEvent("RearTactilTouched", "RearTactil",
                                  "onRearTactilTouched");
  }
  catch (const AL::ALError& e) {
    qiLogError("module.example") << e.what() << std::endl;
  }
}

void RearTactil::onRearTactilTouched() {
  qiLogInfo("module.example") << "Executing callback method on rear tactil pressed event" << std::endl;
  /**
  * As long as this is defined, the code is thread-safe.
  */
  AL::ALCriticalSection section(fCallbackMutex);

  /**
  * Check that the rear tactil is pressed.
  */
  fState =  fMemoryProxy.getData("RearTactilTouched");
  if (fState  > 0.5f) {
    return;
  }
  try {
    fTtsProxy = AL::ALTextToSpeechProxy(getParentBroker());
    // The duration of the video
    int duration = 2;
    AL::ALVideoRecorderProxy videoRecorderProxy(getParentBroker());
    if(videoRecorderProxy.isRecording() != true){
      videoRecorderProxy.setResolution(1);// kVGA : 640*480 
      videoRecorderProxy.setFrameRate(12);//
      std::string filename;
      std::stringstream ss;
      ss << time(0);
      ss >> filename;
      fTtsProxy.say("Action");
      videoRecorderProxy.startRecording("/home/nao/recordings/highquality",filename);
      sleep(duration);
      videoRecorderProxy.stopRecording();
      fTtsProxy.say("End recording");
    }
  }
  catch (const AL::ALError& e) {
    qiLogError("module.example") << e.what() << std::endl;
  }
}

