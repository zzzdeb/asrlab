cmake_minimum_required(VERSION 3.1...3.24)

project(
  asrlab
  VERSION 1.0
  LANGUAGES CXX)

AUX_SOURCE_DIRECTORY(src ASRSOURCES)

add_executable(asrlab ${ASRSOURCES})

target_include_directories(asrlab PUBLIC
  src/rapidjson/include
)

add_library(pgm src/pgm/PGM.cpp)
target_link_libraries(asrlab pgm)

install(TARGETS asrlab RUNTIME DESTINATION ${CMAKE_CURRENT_LIST_DIR})
 