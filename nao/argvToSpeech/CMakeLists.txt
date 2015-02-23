cmake_minimum_required(VERSION 2.8)
project(argvToSpeech)

find_package(qibuild)

# Create a executable named argvToSpeech
# with the source file: main.cpp
qi_create_bin(argvToSpeech "main.cpp")
qi_use_lib(argvToSpeech ALCOMMON)
