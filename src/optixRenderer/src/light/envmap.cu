#include "light/envmap.h"
#include <curand_kernel.h>
#include <ctime>

rtDeclareVariable(int, max_depth, , );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Environmental Lighting 
rtDeclareVariable(int, isEnvmap, , );
rtTextureSampler<float4, 2> envmap0;
rtTextureSampler<float4, 2> envmap1;
rtTextureSampler<float4, 2> envmap2;
rtTextureSampler<float4, 2> envmap3;
rtTextureSampler<float4, 2> envmap4;
rtTextureSampler<float4, 2> envmap5;

rtTextureSampler<float4, 2> envmapDirec0;
rtTextureSampler<float4, 2> envmapDirec1;
rtTextureSampler<float4, 2> envmapDirec2;
rtTextureSampler<float4, 2> envmapDirec3;
rtTextureSampler<float4, 2> envmapDirec4;
rtTextureSampler<float4, 2> envmapDirec5;

rtBuffer<float, 2> envcdfV0;
rtBuffer<float, 2> envcdfV1;
rtBuffer<float, 2> envcdfV2;
rtBuffer<float, 2> envcdfV3;
rtBuffer<float, 2> envcdfV4;
rtBuffer<float, 2> envcdfV5;

rtBuffer<float, 2> envcdfH0;
rtBuffer<float, 2> envcdfH1;
rtBuffer<float, 2> envcdfH2;
rtBuffer<float, 2> envcdfH3;
rtBuffer<float, 2> envcdfH4;
rtBuffer<float, 2> envcdfH5;

rtBuffer<float, 2> envpdf0;
rtBuffer<float, 2> envpdf1;
rtBuffer<float, 2> envpdf2;
rtBuffer<float, 2> envpdf3;
rtBuffer<float, 2> envpdf4;
rtBuffer<float, 2> envpdf5;

rtDeclareVariable(float, infiniteFar, , );


//RT_CALLABLE_PROGRAM float3 EnvUVToDirec(float u, float v){ 
//    // Turn uv coordinate into direction
//    float theta = 2 * (u - 0.5) * M_PIf;
//    float phi = M_PIf * (1 - v); 
//    return make_float3(
//                sinf(phi) * sinf(theta),
//                cosf(phi),
//                sinf(phi) * cosf(theta)
//            );
//}
//
//
//RT_CALLABLE_PROGRAM float2 EnvDirecToUV(const float3& direc){ 
//    float theta = atan2f( direc.x, direc.z );
//    float phi = M_PIf - acosf(direc.y );
//    float u = theta * (0.5f * M_1_PIf) + 0.5;
//    if(u > 1)
//        u = u-1;
//    float v     = phi / M_PIf;
//    return make_float2(u, v);
//}
//
//RT_CALLABLE_PROGRAM float EnvDirecToPdf(const float3& direc) {
//    float2 uv = EnvDirecToUV(direc);
//    size_t2 pdfSize = envpdf.size();
//    float u = uv.x, v = uv.y;
//    int rowId = int(v * (pdfSize.y - 1) );
//    int colId = int(u * (pdfSize.x - 1) );
//    return envpdf[make_uint2(colId, rowId ) ];
//}

RT_CALLABLE_PROGRAM float3 EnvUVToDirec(float u, float v, int face_index) { 
    // Turn uv coordinate into direction
    // convert range 0 to 1 to -1 to 1
    float uc = 2.0f * u - 1.0f;
    float vc = 2.0f * v - 1.0f;
    float x, y, z;
    switch (face_index) {
      case 0: x =  1.0f; y =    vc; z =   -uc; break;	// POSITIVE X
      case 1: x = -1.0f; y =    vc; z =    uc; break;	// NEGATIVE X
      case 2: x =    uc; y =  1.0f; z =   -vc; break;	// POSITIVE Y
      case 3: x =    uc; y = -1.0f; z =    vc; break;	// NEGATIVE Y
      case 4: x =    uc; y =    vc; z =  1.0f; break;	// POSITIVE Z
      case 5: x =   -uc; y =    vc; z = -1.0f; break;	// NEGATIVE Z
    }
    return normalize(make_float3(x, y, z));

}

RT_CALLABLE_PROGRAM float2 EnvDirecToUV(const float3& direc, int& f_index) { 
    float x = direc.x;
    float y = direc.y;
    float z = direc.z;
    
    float absX = fabs(x);
    float absY = fabs(y);
    float absZ = fabs(z);
  
    int isXPositive = x > 0 ? 1 : 0;
    int isYPositive = y > 0 ? 1 : 0;
    int isZPositive = z > 0 ? 1 : 0;
  
    float maxAxis, uc, vc;

    // POSITIVE X
    if (isXPositive && absX >= absY && absX >= absZ) {
      // u (0 to 1) goes from +z to -z
      // v (0 to 1) goes from -y to +y
      maxAxis = absX;
      uc = -z;
      vc = y;
      f_index = 0;
    }
    // NEGATIVE X
    if (!isXPositive && absX >= absY && absX >= absZ) {
      // u (0 to 1) goes from -z to +z
      // v (0 to 1) goes from -y to +y
      maxAxis = absX;
      uc = z;
      vc = y;
      f_index = 1;
    }
    // POSITIVE Y
    if (isYPositive && absY >= absX && absY >= absZ) {
      // u (0 to 1) goes from -x to +x
      // v (0 to 1) goes from +z to -z
      maxAxis = absY;
      uc = x;
      vc = -z;
      f_index = 2;
    }
    // NEGATIVE Y
    if (!isYPositive && absY >= absX && absY >= absZ) {
      // u (0 to 1) goes from -x to +x
      // v (0 to 1) goes from -z to +z
      maxAxis = absY;
      uc = x;
      vc = z;
      f_index = 3;
    }
    // POSITIVE Z
    if (isZPositive && absZ >= absX && absZ >= absY) {
      // u (0 to 1) goes from -x to +x
      // v (0 to 1) goes from -y to +y
      maxAxis = absZ;
      uc = x;
      vc = y;
      f_index = 4;
    }
    // NEGATIVE Z
    if (!isZPositive && absZ >= absX && absZ >= absY) {
      // u (0 to 1) goes from +x to -x
      // v (0 to 1) goes from -y to +y
      maxAxis = absZ;
      uc = -x;
      vc = y;
      f_index = 5;
    }
  
    // Convert range from -1 to 1 to 0 to 1
    auto u = 0.5f * (uc / maxAxis + 1.0f);
    auto v = 0.5f * (vc / maxAxis + 1.0f);
    return make_float2(u, v);
}


RT_CALLABLE_PROGRAM float EnvDirecToPdf(const float3& direc) {
    int f_index;
    float2 uv = EnvDirecToUV(direc, f_index);
    float u = uv.x, v = uv.y; 
    size_t2 pdfSize;
    switch (f_index) {
      case 0: pdfSize = envpdf0.size(); break;
      case 1: pdfSize = envpdf1.size(); break;
      case 2: pdfSize = envpdf2.size(); break;
      case 3: pdfSize = envpdf3.size(); break;
      case 4: pdfSize = envpdf4.size(); break;
      case 5: pdfSize = envpdf5.size(); break;
    }
    int rowId = int(v * (pdfSize.y - 1));
    int colId = int(u * (pdfSize.x - 1));
    switch (f_index) {
      case 0: return envpdf0[make_uint2(colId, rowId ) ]; break;
      case 1: return envpdf1[make_uint2(colId, rowId ) ]; break;
      case 2: return envpdf2[make_uint2(colId, rowId ) ]; break;
      case 3: return envpdf3[make_uint2(colId, rowId ) ]; break;
      case 4: return envpdf4[make_uint2(colId, rowId ) ]; break;
      case 5: return envpdf5[make_uint2(colId, rowId ) ]; break;
    }
}


RT_CALLABLE_PROGRAM void sampleEnvironmapLight(unsigned int& seed, float3& radiance, float3& direction, float& pdfSolidEnv){
    float z1 = rnd(seed);
    float z2 = rnd(seed);
    
    int ncols = envcdfH0.size().x;
    int nrows = envcdfH0.size().y;

    int f_index = (int) (6 * rnd(seed));

    float u = 0, v = 0;
    int rowId = 0;
    int colId = 0;
    // Sample the row 
    switch (f_index) {
        case 0:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV0[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV0[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV0[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV0[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH0[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH0[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH0[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH0[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
    
        case 1:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV1[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV1[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV1[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV1[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH1[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH1[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH1[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH1[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
        case 2:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV2[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV2[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV2[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV2[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH2[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH2[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH2[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH2[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
        case 3:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV3[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV3[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV3[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV3[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH3[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH3[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH3[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH3[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
        case 4:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV4[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV4[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV4[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV4[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH4[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH4[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH4[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH4[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
        case 5:
	    {
		int left = 0, right = nrows-1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfV5[ make_uint2(0, mid) ] >= z1)
			right = mid;
		    else if(envcdfV5[ make_uint2(0, mid) ] < z1)
			left = mid + 1;
		}
		float up = envcdfV5[make_uint2(0, left) ];
		float down = (left == 0) ? 0 : envcdfV5[make_uint2(0, left-1) ];
		v = ( (z1 - down) / fmaxf( (up - down), 1e-14) + left) / float(nrows);
		rowId = left;
	    }

	    // Sample the column
	    {
		int left = 0; int right = ncols - 1;
		while(right > left){
		    int mid = (left + right) / 2;
		    if(envcdfH5[ make_uint2(mid, rowId) ] >= z2)
			right = mid;
		    else if(envcdfH5[ make_uint2(mid, rowId) ] < z2)
			left = mid + 1;
		}
		float up = envcdfH5[make_uint2(left, rowId) ];
		float down = (left == 0) ? 0 : envcdfH5[make_uint2(left-1, rowId) ];
		u = ((z2 - down) / fmaxf((up - down), 1e-14) + left) / float(ncols);
		colId = left;
	    }
	    break;
    }
    // Turn uv coordinate into direction
    //int face_index;
    direction = EnvUVToDirec(u, v, f_index);
    switch (f_index) {
     case 0: pdfSolidEnv = envpdf0[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap0, u, v)); break;
     case 1: pdfSolidEnv = envpdf1[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap1, u, v)); break;
     case 2: pdfSolidEnv = envpdf2[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap2, u, v)); break;
     case 3: pdfSolidEnv = envpdf3[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap3, u, v)); break;
     case 4: pdfSolidEnv = envpdf4[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap4, u, v)); break;
     case 5: pdfSolidEnv = envpdf5[make_uint2(colId,rowId)]; radiance = make_float3(tex2D(envmap5, u, v)); break;
    }
}


RT_PROGRAM void envmap_miss(){
    if(isEnvmap == 0){
        prd_radiance.attenuation = make_float3(0.0);
    }
    else if(isEnvmap == 1){    
	int f_index;
        float2 uv = EnvDirecToUV(prd_radiance.direction, f_index);

        if(prd_radiance.depth == 0){
	    switch (f_index) {
	      case 0: prd_radiance.radiance = make_float3(tex2D(envmapDirec0, uv.x, uv.y) ); break;
	      case 1: prd_radiance.radiance = make_float3(tex2D(envmapDirec1, uv.x, uv.y) ); break;
	      case 2: prd_radiance.radiance = make_float3(tex2D(envmapDirec2, uv.x, uv.y) ); break;
	      case 3: prd_radiance.radiance = make_float3(tex2D(envmapDirec3, uv.x, uv.y) ); break;
	      case 4: prd_radiance.radiance = make_float3(tex2D(envmapDirec4, uv.x, uv.y) ); break;
	      case 5: prd_radiance.radiance = make_float3(tex2D(envmapDirec5, uv.x, uv.y) ); break;
	    }
        }
        else{
	    float3 radiance;
	    switch (f_index) {
	      case 0: radiance = make_float3(tex2D(envmap0, uv.x, uv.y) ); break;
	      case 1: radiance = make_float3(tex2D(envmap1, uv.x, uv.y) ); break;
	      case 2: radiance = make_float3(tex2D(envmap2, uv.x, uv.y) ); break;
	      case 3: radiance = make_float3(tex2D(envmap3, uv.x, uv.y) ); break;
	      case 4: radiance = make_float3(tex2D(envmap4, uv.x, uv.y) ); break;
	      case 5: radiance = make_float3(tex2D(envmap5, uv.x, uv.y) ); break;
	    }
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
