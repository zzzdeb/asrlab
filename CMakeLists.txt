cmake_minimum_required(VERSION 3.1...3.24)

project(
  asrlab
  VERSION 1.0
  LANGUAGES CXX)

AUX_SOURCE_DIRECTORY(src ASRSOURCES)

add_library(asrlablib ${ASRSOURCES})

target_include_directories(asrlablib PUBLIC
  src/rapidjson/include
)

add_library(pgm
  src/pgm/PGM.cpp
  src/pgm/Matrix.cpp
  src/pgm/Section.cpp
)
target_link_libraries(asrlablib pgm)

add_executable(asrlab src/main.cpp)
target_link_libraries(asrlab asrlablib)

install(TARGETS asrlab RUNTIME DESTINATION ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(src/test)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 --std=c++14 -pedantic -msse -msse2 -msse3 -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O2 --std=c++14 -pedantic -msse -msse2 -msse3 -flto")
#SET(CMAKE_AR  "gcc-ar")
#SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
#SET(CMAKE_CXX_ARCHIVE_FINISH   true)
