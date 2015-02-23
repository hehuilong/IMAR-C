#include <iostream>
#include <alerror/alerror.h>
#include <alproxies/altexttospeechproxy.h>
#include <alproxies/almemoryproxy.h>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>

#include <qi/log.hpp>
#include <stdlib.h>
#include <vector>
#include <string>
void helloAnimation(std::string robotIP);

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cerr << "Usage: argvToSpeech <Robot IP> <Phrase> <with> <several> <words>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string name(argv[2]);
    std::string phraseToSay(argv[2]);
    for(int i=3 ; i<argc ; i++){
        phraseToSay = phraseToSay + " " + argv[i];
    }
    AL::ALMemoryProxy  fMemoryProxy(argv[1]);
    AL::ALTextToSpeechProxy ttsp(argv[1],9559);
    ttsp.setLanguage("French");
    ttsp.say(phraseToSay);

    return EXIT_SUCCESS;
}
