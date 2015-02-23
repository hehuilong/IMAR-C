cmake_minimum_required(VERSION 2.8)
project(visionaction)

find_package(qibuild)

# Root directory
set(ROOTDIR ../..)

# Source code, include and libraries directories
set(SRCDIR ${ROOTDIR}/src)
set(INCLUDEDIR ${ROOTDIR}/include)
# set(LIBDIR ${ROOTDIR}/lib)

include_directories(${INCLUDEDIR}/kmlocal ${INCLUDEDIR}/densetrack ${INCLUDEDIR} /usr/local/include)

set(_srcs
    ${SRCDIR}/IplImageWrapper.cpp
    ${SRCDIR}/IplImagePyramid.cpp
    ${SRCDIR}/naodensetrack.cpp
    ${SRCDIR}/naokmeans.cpp
    ${SRCDIR}/naosvm.cpp
    ${SRCDIR}/naomngt.cpp
    ${SRCDIR}/imconfig.cpp
    ${SRCDIR}/imbdd.cpp
    integration.cpp
    tactil.cpp
    )

qi_create_lib(visionaction SHARED ${_srcs} "main.cpp")
qi_use_lib(visionaction ALCOMMON OPENCV2_CORE OPENCV2_HIGHGUI OPENCV2_IMGPROC OPENCV2_VIDEO)
# Searching for shared libraries
target_link_libraries (visionaction libsvm.so libkmeans.so)
