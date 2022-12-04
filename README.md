# Group 1

## Setup:

1. Set up C++ compiler
2. Clone repository **including** submodules:
```
git clone https://git.rwth-aachen.de/asrlabws2223/group-1.git --recurse-submodules
```
3. Possibly need to install additional libraries such as libblas and libgsl:
```
sudo apt-get install libblas-dev libgsl-dev libboost-all-dev
```
or equivalent.
to compile test Boost 1.74 needed. `sudo apt-get install libboost-all-dev`
4. Compile code:
  - Using the makefile:
  ```bash
  cd src/
  make
  cd ..
  ```
  - Using cmake
  ```bash
  mkdir build && cd build
  cmake ..
  make install
  cd ..
  ```
5. Run program using the asrlab executable and a config file:
```
./asrlab config/extract_wsj.config
```
Expected output:
```
Processing (1): example-wsj-1
```
