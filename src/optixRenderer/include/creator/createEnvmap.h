#ifndef CREATEENVMAP_HEADER
#define CREATEENVMAP_HEADER

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "lightStructs.h"
#include "inout/rgbe.h"
#include "creator/createTextureSampler.h"
#include "structs/constant.h"

using namespace optix;


std::array<cv::Mat, 6> loadEnvmap(Envmap env, unsigned* width, unsigned* height);

void createEnvmapBuffer(Context& context, 
        std::array<cv::Mat, 6>& envMat, std::array<cv::Mat, 6>& envMatBlured, 
        unsigned gridWidth = 0, unsigned gridHeight = 0);

void computeEnvmapDistribution(
            Context& context,
            std::array<cv::Mat, 6> envMat,
            unsigned width = 1024, unsigned height = 512, 
            unsigned gridWidth = 0, unsigned gridHeight = 0);

void createEnvmap(
        Context& context,
        std::vector<Envmap>& envmaps, 
        unsigned width = 1024, unsigned height = 512, 
        unsigned gridWidth = 0, unsigned gridHeight = 0 );

void rotateUpdateEnvmap(Context& context, Envmap& env, float phiDelta, 
        unsigned width = 1024, unsigned height = 512,
        unsigned gridWidth = 1024, unsigned gridHeight = 512 );


#endif
