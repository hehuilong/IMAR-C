#ifndef TACTIL_H
#define TACTIL_H

#include <string>
#include <boost/shared_ptr.hpp>

#include <qi/log.hpp>

#include <alvalue/alvalue.h>
#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <alcommon/almodule.h>

#include <althread/alcriticalsection.h>
#include <althread/almutex.h>

#include <alerror/alerror.h>

#include <alproxies/almemoryproxy.h>
#include <alproxies/altexttospeechproxy.h>
#include <alproxies/alrobotpostureproxy.h>
#include <alproxies/almotionproxy.h>
#include <alproxies/alvideorecorderproxy.h>
#include <alproxies/alledsproxy.h>

#include "integration.h"

namespace AL
{
  class ALBroker;
}

class Tactil : public AL::ALModule
{
  public:

   Tactil(boost::shared_ptr<AL::ALBroker> broker, const std::string& name);

    virtual ~Tactil();

    /** Overloading ALModule::init().
    * This is called right after the module has been loaded
    */
    virtual void init();

    /**
    * This method will be called every time the event FrontTactilTouched is raised.
    */
    void onFrontHeadTouched();

    void helloAnimation();
    void applaudAnimation();

    void decrease_quality(std::string videoInput, std::string videoOutput);
  private:
    AL::ALMemoryProxy fMemoryProxy;
    AL::ALTextToSpeechProxy ttsp;
    AL::ALRobotPostureProxy rpp;
    AL::ALVideoRecorderProxy videoRecorderProxy;
    AL::ALLedsProxy lp;
    boost::shared_ptr<AL::ALMutex> fCallbackMutex;
    float fState;

};
#endif // TACTIL_H
