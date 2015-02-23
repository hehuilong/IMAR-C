cmake_minimum_required(VERSION 2.8)
project(reartactil)
find_package(qibuild)

option(BUMPER_IS_REMOTE
  "module is compiled as a remote module (ON or OFF)"
  OFF)

set(_srcs
  main.cpp
  reartactil.h
  reartactil.cpp
)

if(BUMPER_IS_REMOTE)
  add_definitions(" -DBUMPER_IS_REMOTE ")
  qi_create_bin(reartactil ${_srcs})
else()
  qi_create_lib(reartactil SHARED ${_srcs} SUBFOLDER naoqi)
endif()

qi_use_lib(reartactil ALCOMMON)
