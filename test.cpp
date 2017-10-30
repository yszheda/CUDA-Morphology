#include <cstdio>
#include <iostream>
#include <random>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "morphology.h"


int main(int argc, char* argv[])
{
    cv::Mat srcImg;
    int hsize = 3;
    int vsize = 3;

    if (argc > 1) {
        srcImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    } else {
        std::cout << "./test <image name> [hsize of Sel] [vsize of Sel]" << std::endl;
        return 1;
    }

    if (argc > 2) {
        hsize = atoi(argv[2]);
    }

    if (argc > 3) {
        vsize = atoi(argv[3]);
    }

    int width = srcImg.cols;
    int height = srcImg.rows;
    auto memSize = width * height * sizeof(unsigned char);
    unsigned char* img_d;
    cudaMalloc((void **) &img_d, memSize);

    cudaMemcpy(img_d, srcImg.data, memSize, cudaMemcpyHostToDevice);
    erode(img_d, width, height, hsize, vsize);
    cudaMemcpy(srcImg.data, img_d, memSize, cudaMemcpyDeviceToHost);

    cv::imwrite("erode.jpg", srcImg);

    cudaMemcpy(img_d, srcImg.data, memSize, cudaMemcpyHostToDevice);
    dilate(img_d, width, height, hsize, vsize);
    cudaMemcpy(srcImg.data, img_d, memSize, cudaMemcpyDeviceToHost);

    cv::imwrite("dilate.jpg", srcImg);

    cudaFree(img_d);

    return 0;
}
