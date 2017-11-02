# CUDA-Morphology

Morphology erode/dilate in GPU using CUDA

## Compile static library and test program

1. Install CUDA
2. Install OpenCV
3. Install CMake

```
mkdir build
cd build
cmake ..
make
```

## Run test program

```
./test <image name> [<hsize of Sel>] [<vsize of Sel>]
```
