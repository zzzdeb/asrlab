# Group 1

## Setup:

1. Set up C++ compiler
2. Clone repository **including** submodules:
```
git clone https://git.rwth-aachen.de/asrlabws2223/group-1.git --recurse-submodules
```
3. Possibly need to install additional libraries such as libblas and libgsl:
```
sudo apt-get install libblas-dev libgsl-dev
```
or equivalent
4. Compile code using the makefile:
```
cd src/
make
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
