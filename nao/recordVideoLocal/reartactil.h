/**
 * @author HE Huilong 
 *
 * modified by HE Huilong (02/09/2013), change the event to rear tactil pressed 
 */

#ifndef BUMPER_BUMPER_H
#define BUMPER_BUMPER_H

#include <boost/shared_ptr.hpp>
#include <alcommon/almodule.h>
#include <string>

#include <alproxies/almemoryproxy.h>
#include <alproxies/altexttospeechproxy.h>
#include <althread/almutex.h>

namespace AL
{
  class ALBroker;
}

class RearTactil : public AL::ALModule
{
  public:

    RearTactil(boost::shared_ptr<AL::ALBroker> broker, const std::string& name);

    virtual ~RearTactil();

    /** Overloading ALModule::init().
    * This is called right after the module has been loaded
    */
    virtual void init();

    /**
    * This method will be called every time the event RearTactilTouched is raised.
    */
    void onRearTactilTouched();

  private:
    AL::ALMemoryProxy fMemoryProxy;
    AL::ALTextToSpeechProxy fTtsProxy;

    boost::shared_ptr<AL::ALMutex> fCallbackMutex;

    float fState;

};

#endif  // BUMPER_BUMPER_H
