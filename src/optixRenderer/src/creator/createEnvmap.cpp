#include "creator/createEnvmap.h"

std::array<cv::Mat, 6> loadEnvmap(Envmap env, unsigned* width, unsigned* height){
    
    int envWidth, envHeight;
    FILE* hdrRead = fopen(env.fileName.c_str(), "r");
    int rgbeInd = RGBE_ReadHeader( hdrRead, &envWidth, &envHeight, NULL);

    if( rgbeInd == -1 ){
        std::cout<<"Wrong: fail to load hdr image."<<std::endl;
        exit(0);
    }
    else{
        float* hdr = new float [envHeight * envWidth * 3];
        for(int n = 0; n < envHeight * envWidth * 3; n++)
            hdr[n] = 0;
        RGBE_ReadPixels_RLE(hdrRead, hdr, envWidth, envHeight );
        fclose(hdrRead );

        // Resize the  image
        cv::Mat envMat(envHeight, envWidth, CV_32FC3);
        for(int r = 0; r < envHeight; r++) {
            for(int c = 0; c < envWidth; c++) {
                int hdrInd = 3 *(r*envWidth + c );
                for(int ch = 0; ch < 3; ch++) {
                    float color = hdr[hdrInd + ch];
                    envMat.at<cv::Vec3f>(r, envWidth -1 - c)[2 - ch] = color * env.scale;
                } 
            }
        }
	int face_dim = envHeight / 3;
	std::array<cv::Mat, 6> envMatFaces;

        envMatFaces[1] = envMat(cv::Rect(0*face_dim, 1*face_dim, face_dim, face_dim));
        envMatFaces[2] = envMat(cv::Rect(1*face_dim, 0*face_dim, face_dim, face_dim));
        envMatFaces[4] = envMat(cv::Rect(1*face_dim, 1*face_dim, face_dim, face_dim));
        envMatFaces[3] = envMat(cv::Rect(1*face_dim, 2*face_dim, face_dim, face_dim));
        envMatFaces[0] = envMat(cv::Rect(2*face_dim, 1*face_dim, face_dim, face_dim));
        envMatFaces[5] = envMat(cv::Rect(3*face_dim, 1*face_dim, face_dim, face_dim));


	*height = face_dim;
	*width = face_dim;
        //cv::Mat envMatNew(height, width, CV_32FC3);
        //cv::resize(envMat, envMatNew, cv::Size(width, height), cv::INTER_AREA); 
        
        delete [] hdr;

        return envMatFaces;
    }
}


void createEnvmapBuffer(Context& context, 
        std::array<cv::Mat, 6>& envMat, std::array<cv::Mat, 6>& envMatBlured, 
        unsigned gridWidth, unsigned gridHeight )
{
    unsigned int width = envMat[0].cols;
    unsigned int height = envMat[0].rows;
    gridWidth = (gridWidth == 0) ? width : gridWidth;
    gridHeight = (gridHeight == 0) ? height : gridHeight;

    // Create tex sampler and populate with default values
    TextureSampler envmapSampler[6];
    TextureSampler envmapDirectSampler[6];
    Buffer envcdfV[6]; 
    Buffer envcdfH[6]; 
    Buffer envpdf[6]; 
    for (int i = 0; i < 6; i++) {
        envmapSampler[i] = createTextureSampler(context);
	envmapDirectSampler[i] = createTextureSampler(context);

        loadImageToTextureSampler(context, envmapSampler[i], envMatBlured[i] );
        loadImageToTextureSampler(context, envmapDirectSampler[i], envMat[i] );

	envcdfV[i] = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1, gridHeight);
	envcdfH[i] = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, gridWidth, gridHeight);
	envpdf[i] = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, gridWidth, gridHeight);
    }

    //curandState_t state;
    //curand_init(time(0), 0, 0, &state);
    
    context["envmap0"] -> setTextureSampler(envmapSampler[0]);
    context["envmap1"] -> setTextureSampler(envmapSampler[1]);
    context["envmap2"] -> setTextureSampler(envmapSampler[2]);
    context["envmap3"] -> setTextureSampler(envmapSampler[3]);
    context["envmap4"] -> setTextureSampler(envmapSampler[4]);
    context["envmap5"] -> setTextureSampler(envmapSampler[5]);

    context["envmapDirec0"] -> setTextureSampler(envmapDirectSampler[0]);
    context["envmapDirec1"] -> setTextureSampler(envmapDirectSampler[1]);
    context["envmapDirec2"] -> setTextureSampler(envmapDirectSampler[2]);
    context["envmapDirec3"] -> setTextureSampler(envmapDirectSampler[3]);
    context["envmapDirec4"] -> setTextureSampler(envmapDirectSampler[4]);
    context["envmapDirec5"] -> setTextureSampler(envmapDirectSampler[5]);
  
    context["envcdfV0"] -> setBuffer(envcdfV[0]);
    context["envcdfV1"] -> setBuffer(envcdfV[1]);
    context["envcdfV2"] -> setBuffer(envcdfV[2]);
    context["envcdfV3"] -> setBuffer(envcdfV[3]);
    context["envcdfV4"] -> setBuffer(envcdfV[4]);
    context["envcdfV5"] -> setBuffer(envcdfV[5]);

    context["envcdfH0"] -> setBuffer(envcdfH[0]);
    context["envcdfH1"] -> setBuffer(envcdfH[1]);
    context["envcdfH2"] -> setBuffer(envcdfH[2]);
    context["envcdfH3"] -> setBuffer(envcdfH[3]);
    context["envcdfH4"] -> setBuffer(envcdfH[4]);
    context["envcdfH5"] -> setBuffer(envcdfH[5]);

    context["envpdf0"] -> setBuffer(envpdf[0]);
    context["envpdf1"] -> setBuffer(envpdf[1]);
    context["envpdf2"] -> setBuffer(envpdf[2]);
    context["envpdf3"] -> setBuffer(envpdf[3]);
    context["envpdf4"] -> setBuffer(envpdf[4]);
    context["envpdf5"] -> setBuffer(envpdf[5]);
}

void computeEnvmapDistribution(
            Context& context,
            std::array<cv::Mat, 6> envMat,
            unsigned width, unsigned height, 
            unsigned gridWidth, unsigned gridHeight
        )
{
    gridWidth = (gridWidth == 0) ? width : gridWidth;
    gridHeight = (gridHeight == 0) ? height : gridHeight;
    printf("width: %d, height: %d\n", gridWidth, gridHeight);

    for (int face = 0; face < 6; face++) {
        optix::Buffer envcdfV = context["envcdfV" + std::to_string(face)] -> getBuffer();
        optix::Buffer envcdfH = context["envcdfH" + std::to_string(face)] -> getBuffer();
        optix::Buffer envpdf = context["envpdf" + std::to_string(face)] -> getBuffer(); 
        // Compute the weight map
        float* envWeight = new float[gridWidth * gridHeight];
        int winH = int(height / gridHeight);
        int winW = int(width  / gridWidth);
        for(int hid = 0; hid < gridHeight; hid++){
            float hidf = float(hid + 0.5) / float(gridHeight);
            float theta = hidf * PI;
            for(int wid = 0; wid < gridWidth; wid++){
                int hs = hid * winH;
                int he = (hid == gridHeight-1) ? height : (hid + 1) * winH;
                int ws = wid * winW;
                int we = (wid == gridWidth-1) ? width : (wid + 1) * winW;
                
                float N = (he - hs) * (we - ws);
                float W = 0;
                for(int i = hs; i < he; i++){
                    for(int j = ws; j < we; j++){
                        for(int ch = 0; ch < 3; ch++)
                            W += envMat[face].at<cv::Vec3f>(height - 1 - i, j)[ch] / 3.0;
                    }
                }
                envWeight[hid*gridWidth + wid] = W / N * sinf(theta) + 1e-10;   
            }
        }

        // Compute the cdf of envmap 
        float* envcdfV_p = reinterpret_cast<float*>(envcdfV -> map() );
        float* envcdfH_p = reinterpret_cast<float*>(envcdfH -> map() );  
        for(int vId = 0; vId < gridHeight; vId++){
            int offset = vId * gridWidth;
            envcdfH_p[offset] = envWeight[offset];
            for(int uId = 1; uId < gridWidth; uId++){
                envcdfH_p[offset + uId] = 
                envcdfH_p[offset + uId-1] + envWeight[offset + uId];
            }
            float rowSum = envcdfH_p[offset + gridWidth-1];
            if(vId == 0)
                envcdfV_p[vId] = rowSum;
            else
                envcdfV_p[vId] = envcdfV_p[vId-1] + rowSum;
            for(int uId = 0; uId < gridWidth; uId++){
                float S = (rowSum > 1e-6) ? rowSum : 1e-6;
                envcdfH_p[offset + uId] /= S;
            }
        }
        float colSum = envcdfV_p[gridHeight-1];
        for(int vId = 0; vId < gridHeight; vId++){
            float S = (colSum > 1e-6) ? colSum : 1e-6;
            envcdfV_p[vId] /= colSum;
        }

        // Compute the pdf for sampling
        int gridSize = gridHeight * gridWidth;
        for(int vId = 0; vId < gridHeight; vId++){
            float vIdf = float(vId + 0.5) / float(gridHeight);
            float sinTheta = sinf(vIdf * PI);
            for(int uId = 0; uId < gridWidth; uId++){ 
                float deno = ((colSum * 2 * PI * PI * sinTheta) / gridSize );
                envWeight[vId * gridWidth+ uId] /= ((deno > 1e-6) ? deno : 1e-6);
            }
        }
    
        float* envpdf_p = static_cast<float*> (envpdf -> map() );
        for(unsigned int i = 0; i < gridHeight; i++){
            for(unsigned int j = 0; j < gridWidth; j++){            
                int envId = i * gridWidth + j;
                envpdf_p[envId] = envWeight[envId];
            }
        }
        envpdf -> unmap();
        envcdfV -> unmap();
        envcdfH -> unmap();
        delete [] envWeight;
    }




}


void createEnvmap(
        Context& context,
        std::vector<Envmap>& envmaps, 
        unsigned width, unsigned height, 
        unsigned gridWidth, unsigned gridHeight
        )
{
    if(gridWidth == 0 || gridHeight == 0){
        gridWidth = width;
        gridHeight = height;
    }

    if(envmaps.size() == 0){
        context["isEnvmap"] -> setInt(0);
        // Create the texture sampler 
	std::array<cv::Mat, 6> emptyMat;
	for (int i = 0; i < 6; i++) {
            emptyMat[i] = cv::Mat::zeros(1, 1, CV_32FC3);
	}
        createEnvmapBuffer(context, emptyMat, emptyMat, 1, 1);
    }
    else{
        // Load the environmental map
        Envmap env = envmaps[0];
	unsigned envwidth, envheight;
	std::array<cv::Mat, 6> envMat = loadEnvmap(env, &envwidth, &envheight);
	printf("envMatFaces %f\n", envMat[0].at<cv::Vec3f>(1, 3)[0]);
	if(gridWidth == width || gridHeight == height){
	    gridWidth = envwidth;
	    gridHeight = envheight;
	}
	    printf("ewidth: %d, eheight: %d\n", envwidth, envheight);
	    printf("gwidth: %d, gheight: %d\n", gridWidth, gridHeight);
        
        unsigned kernelSize = std::max(5, int(envheight / 100) );
        if(kernelSize % 2 == 0) kernelSize += 1;
	std::array<cv::Mat, 6> envMatBlured;
	for (int i = 0; i < 6; i++) {
            envMatBlured[i] = cv::Mat(envheight, envwidth, CV_32FC3);
            cv::GaussianBlur(envMat[i], envMatBlured[i], cv::Size(kernelSize, kernelSize), 0, 0); 
	}

        context["isEnvmap"] -> setInt(1); 
        createEnvmapBuffer(context, envMat, envMatBlured, gridWidth, gridHeight);

        computeEnvmapDistribution(context, envMatBlured, 
                envwidth, envheight, gridWidth, gridHeight);
	printf("Finished\n");
    }
}

//void rotateUpdateEnvmap(Context& context, Envmap& env, float phiDelta, 
//        unsigned width, unsigned height,
//        unsigned gridWidth, unsigned gridHeight
//        )
//{
//    cv::Mat envMat = loadEnvmap(env, &width, &height);
//    cv::Mat envMatNew(height, width, CV_32FC3);
//    
//    // Rotate the envmap  
//    for(int r = 0; r < height; r++){
//        for(int c = 0; c < width; c++){
//            float phi = (2 * float(c) / width - 1) * PI;
//            float phiNew = phi + phiDelta;
//            while(phiNew >= PI) phiNew -= 2 * PI;
//            while(phiNew < -PI) phiNew += 2 * PI;
//
//            float cNew = (float(width) -1) * (phiNew + PI) / (2 * PI);
//            int c1 = int(ceil(cNew ) );
//            int c2 = int(floor(cNew ) );
//            float w1 = cNew - c2;
//            float w2 = (c1 == c2) ? 0 : c1 - cNew;
//            
//            for(int ch = 0; ch < 3; ch++){
//                envMatNew.at<cv::Vec3f>(r, c)[ch] = 
//                    w1 * envMat.at<cv::Vec3f>(r, c1)[ch]
//                    + w2 * envMat.at<cv::Vec3f>(r, c2)[ch];
//            }
//        }
//    }
//    
//    unsigned kernelSize = std::max(3, int(height / 200) );
//    if(kernelSize % 2 == 0) kernelSize += 1;
//    cv::Mat envMatNewBlured(height, width, CV_32FC3);
//    cv::GaussianBlur(envMatNew, envMatNewBlured, cv::Size(kernelSize, kernelSize), 0, 0); 
//
//    // Update the texture 
//    TextureSampler Sampler = context["envmap"] -> getTextureSampler();
//    updateImageToTextureSampler(Sampler, envMatNewBlured);
//    computeEnvmapDistribution(context, envMatNewBlured, width, height, gridWidth, gridHeight); 
//     
//    TextureSampler direcSampler = context["envmapDirect"] -> getTextureSampler();
//    updateImageToTextureSampler(direcSampler, envMatNew);
//}
