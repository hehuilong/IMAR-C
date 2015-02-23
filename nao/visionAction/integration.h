#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "naosvm.h"
#include "naokmeans.h"
#include "naodensetrack.h"
#include "naomngt.h"

#include "Box.h"
#include <iostream>
#include <fstream>
#include <qi/log.hpp>
#include <sys/times.h>
typedef struct tms sTms;
typedef struct{
  int begin,end;
  sTms sbegin, send;
} exec_time;
void printTime(exec_time *tmps);
std::string integration(std::string video, std::string folder);
#endif // INTEGRATION_H
