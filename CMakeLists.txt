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

set (Boost_USE_STATIC_LIBS OFF)
find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})
add_executable (asrlabtest src/test/test2.cpp)
target_link_libraries(asrlabtest
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      asrlablib
                      )
