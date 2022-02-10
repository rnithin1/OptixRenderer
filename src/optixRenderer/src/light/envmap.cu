#include "light/envmap.h"

rtDeclareVariable(int, max_depth, , );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Environmental Lighting 
rtDeclareVariable(int, isEnvmap, , );
rtTextureSampler<float4, 2> envmap;
rtTextureSampler<float4, 2> envmapDirec;
rtBuffer<float, 2> envcdfV;
rtBuffer<float, 2> envcdfH;
rtBuffer<float, 2> envpdf;
rtDeclareVariable(float, infiniteFar, , );


RT_CALLABLE_PROGRAM float3 EnvUVToDirec(float u, float v){ 
    // Turn uv coordinate into direction
    float theta = 2 * (u - 0.5) * M_PIf;
    float phi = M_PIf * (1 - v); 
    return make_float3(
                sinf(phi) * sinf(theta),
                cosf(phi),
                sinf(phi) * cosf(theta)
            );
}


RT_CALLABLE_PROGRAM float2 EnvDirecToUV(const float3& direc){ 
    float theta = atan2f( direc.x, direc.z );
    float phi = M_PIf - acosf(direc.y );
    float u = theta * (0.5f * M_1_PIf) + 0.5;
    if(u > 1)
        u = u-1;
    float v     = phi / M_PIf;
    return make_float2(u, v);
}

RT_CALLABLE_PROGRAM float EnvDirecToPdf(const float3& direc) {
    float2 uv = EnvDirecToUV(direc);
    size_t2 pdfSize = envpdf.size();
    float u = uv.x, v = uv.y;
    int rowId = int(v * (pdfSize.y - 1) );
    int colId = int(u * (pdfSize.x - 1) );
    return envpdf[make_uint2(colId, rowId ) ];
}

//RT_CALLABLE_PROGRAM float3 EnvUVToDirec(float u, float v, int* index) { 
//    // Turn uv coordinate into direction
//    // convert range 0 to 1 to -1 to 1
//    float uc = 2.0f * u - 1.0f;
//    float vc = 2.0f * v - 1.0f;
//    float x, y, z;
//    switch (index) {
//      case 0: x =  1.0f; y =    vc; z =   -uc; break;	// POSITIVE X
//      case 1: x = -1.0f; y =    vc; z =    uc; break;	// NEGATIVE X
//      case 2: x =    uc; y =  1.0f; z =   -vc; break;	// POSITIVE Y
//      case 3: x =    uc; y = -1.0f; z =    vc; break;	// NEGATIVE Y
//      case 4: x =    uc; y =    vc; z =  1.0f; break;	// POSITIVE Z
//      case 5: x =   -uc; y =    vc; z = -1.0f; break;	// NEGATIVE Z
//    }
//    return normalize(make_float3(x, y, z));
//
//}
//
//RT_CALLABLE_PROGRAM float2 EnvDirecToUV(const float3& direc, int* index) { 
//    float x = direc.x;
//    float y = direc.y;
//    float z = direc.z;
//    
//    float absX = fabs(x);
//    float absY = fabs(y);
//    float absZ = fabs(z);
//  
//    int isXPositive = x > 0 ? 1 : 0;
//    int isYPositive = y > 0 ? 1 : 0;
//    int isZPositive = z > 0 ? 1 : 0;
//  
//    float maxAxis, uc, vc;
//  
//    // POSITIVE X
//    if (isXPositive && absX >= absY && absX >= absZ) {
//      // u (0 to 1) goes from +z to -z
//      // v (0 to 1) goes from -y to +y
//      maxAxis = absX;
//      uc = -z;
//      vc = y;
//      *index = 0;
//    }
//    // NEGATIVE X
//    if (!isXPositive && absX >= absY && absX >= absZ) {
//      // u (0 to 1) goes from -z to +z
//      // v (0 to 1) goes from -y to +y
//      maxAxis = absX;
//      uc = z;
//      vc = y;
//      *index = 1;
//    }
//    // POSITIVE Y
//    if (isYPositive && absY >= absX && absY >= absZ) {
//      // u (0 to 1) goes from -x to +x
//      // v (0 to 1) goes from +z to -z
//      maxAxis = absY;
//      uc = x;
//      vc = -z;
//      *index = 2;
//    }
//    // NEGATIVE Y
//    if (!isYPositive && absY >= absX && absY >= absZ) {
//      // u (0 to 1) goes from -x to +x
//      // v (0 to 1) goes from -z to +z
//      maxAxis = absY;
//      uc = x;
//      vc = z;
//      *index = 3;
//    }
//    // POSITIVE Z
//    if (isZPositive && absZ >= absX && absZ >= absY) {
//      // u (0 to 1) goes from -x to +x
//      // v (0 to 1) goes from -y to +y
//      maxAxis = absZ;
//      uc = x;
//      vc = y;
//      *index = 4;
//    }
//    // NEGATIVE Z
//    if (!isZPositive && absZ >= absX && absZ >= absY) {
//      // u (0 to 1) goes from +x to -x
//      // v (0 to 1) goes from -y to +y
//      maxAxis = absZ;
//      uc = -x;
//      vc = y;
//      *index = 5;
//    }
//  
//    // Convert range from -1 to 1 to 0 to 1
//    u = 0.5f * (uc / maxAxis + 1.0f);
//    v = 0.5f * (vc / maxAxis + 1.0f);
//    return make_float2(u, v);
//}
//
//
//RT_CALLABLE_PROGRAM float EnvDirecToPdf(const float3& direc) {
//    float2 uv = EnvDirecToUV(direc);
//    size_t2 pdfSize = envpdf.size();
//    float u = uv.x, v = uv.y;
//    int rowId = int(v * (pdfSize.y - 1) );
//    int colId = int(u * (pdfSize.x - 1) );
//    return envpdf[make_uint2(colId, rowId ) ];
//}


RT_CALLABLE_PROGRAM void sampleEnvironmapLight(unsigned int& seed, float3& radiance, float3& direction, float& pdfSolidEnv){
    float z1 = rnd(seed);
    float z2 = rnd(seed);
    
    int ncols = envcdfH.size().x;
    int nrows = envcdfH.size().y;

    // Sample the row 
    float u = 0, v = 0;
    int rowId = 0;
    {
        int left = 0, right = nrows-1;
        while(right > left){
            int mid = (left + right) / 2;
            if(envcdfV[ make_uint2(0, mid) ] >= z1)
                right = mid;
            else if(envcdfV[ make_uint2(0, mid) ] < z1)
                left = mid + 1;
        }
        float up = envcdfV[make_uint2(0, left) ];
        float down = (left == 0) ? 0 : envcdfV[make_uint2(0, left-1) ];
        v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
        rowId = left;
    }

    // Sample the column
    int colId = 0;
    {
        int left = 0; int right = ncols - 1;
        while(right > left){
            int mid = (left + right) / 2;
            if(envcdfH[ make_uint2(mid, rowId) ] >= z2)
                right = mid;
            else if(envcdfH[ make_uint2(mid, rowId) ] < z2)
                left = mid + 1;
        }
        float up = envcdfH[make_uint2(left, rowId) ];
        float down = (left == 0) ? 0 : envcdfH[make_uint2(left-1, rowId) ];
        u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
        colId = left;
    }
    
    // Turn uv coordinate into direction
    direction = EnvUVToDirec(u, v);
    pdfSolidEnv = envpdf[make_uint2(colId, rowId) ];
    radiance = make_float3(tex2D(envmap, u, v) );
}


RT_PROGRAM void envmap_miss(){
    if(isEnvmap == 0){
        prd_radiance.attenuation = make_float3(0.0);
    }
    else if(isEnvmap == 1){    
        float2 uv = EnvDirecToUV(prd_radiance.direction);

        if(prd_radiance.depth == 0){
            prd_radiance.radiance = make_float3(tex2D(envmapDirec, uv.x, uv.y) ); 
        }
        else{
            float3 radiance = make_float3(tex2D(envmap, uv.x, uv.y) );
            // Multiple Importance Sampling 
            if(prd_radiance.pdf < 0){
                prd_radiance.radiance += radiance * prd_radiance.attenuation;
            }
            else{
                float pdfSolidEnv = EnvDirecToPdf(prd_radiance.direction);
                float pdfSolidBRDF = prd_radiance.pdf;
                float pdfSolidEnv2 = pdfSolidEnv * pdfSolidEnv;
                float pdfSolidBRDF2 = pdfSolidBRDF * pdfSolidBRDF;

                float3 radianceInc = radiance  * pdfSolidBRDF2 / fmaxf(pdfSolidBRDF2 + pdfSolidEnv2, 1e-14)* prd_radiance.attenuation;
                prd_radiance.radiance += radianceInc;
            }
        }
    }
    prd_radiance.done = true;
}


RT_PROGRAM void miss(){
    prd_radiance.radiance = make_float3(0.0);
    prd_radiance.done = true;
}
