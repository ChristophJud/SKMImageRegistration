/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Benedikt Bitterli (benedikt.bitterli@unibas.ch)
 *         Christoph Jud     (christoph.jud@unibas.ch)
 *         Robin Sandkühler  (robin.sandkuehler@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GpuEvaluator.h"

struct GpuLCCParams
{
    ScalarType *ccvalues;
};

template<typename Type, int Dimensions>
__host__ __device__ inline Type neighborhoodWeight(
    const Vec<Type,Dimensions>& p, const Vec<Type,Dimensions>& q, float support, float beta)
{
    const Type r = pow( (p - q).length()/support, beta );
    const Type f = max(Type(0.0), Type(1.0) - r);
    return f*f;
}

#if SpaceDimensions == 3
template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateValueLCC(GpuParams params, GpuLCCParams lcc_params)
{
    Vec3i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3i samplePos = Vec3i(x, y, z);
    if (DoSubsample)
        samplePos = samplePos*params.subsample + params.gridShift[x + y*size[0] + z*size[0]*size[1]];
    Vec3f  fixedImagePoint = params.fixedImage.toGlobal(Vec3f(samplePos));
    Vec3f centerRegionPoint = fixedImagePoint;
    if (params.useDisplacementField)
        centerRegionPoint += params.displacementField.atGlobal(fixedImagePoint);

    ScalarType support;
    ScalarType sigmaP;
    Mat3f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec3f indexPoint = params.cpImage.toLocal(fixedImagePoint);
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, covP);
                else
                    k = params.kernel.evaluate(fixedImagePoint, point_i);

                centerRegionPoint += params.cpwImage.at(Vec3i(xi, yi, zi))*k*params.cpImage.at(Vec3i(xi, yi, zi));
            }
        }
    }

    ScalarType ccvalue=0.0;


    // TODO: make subsampled fixed image region as in gather gradients
        Vec3f fIndexPoint = params.fixedImage.toLocal(fixedImagePoint);
        Vec3i fImgLower = std::min(std::max(Vec3i(fIndexPoint - support/params.fixedImage.scale() + 1.0), Vec3i(0)), params.fixedImage.size() - 1);
        Vec3i fImgUpper = std::min(std::max(Vec3i(fIndexPoint + support/params.fixedImage.scale()      ), Vec3i(0)), params.fixedImage.size() - 1);

        ScalarType sf    = 0.0;
        ScalarType sm    = 0.0;
        ScalarType sff   = 0.0;
        ScalarType smm   = 0.0;
        ScalarType sfm   = 0.0;
        int insideRegion = 0;
        for (int fzi = fImgLower[2]; fzi <= fImgUpper[2]; fzi+=params.subsampleNeighborhood) {  
            for (int fyi = fImgLower[1]; fyi <= fImgUpper[1]; fyi+=params.subsampleNeighborhood) {      // loop over fixed image kernel region
                for (int fxi = fImgLower[0]; fxi <= fImgUpper[0]; fxi+=params.subsampleNeighborhood) {

                    // TODO: do subsampling as in gatherGradients
                    Vec3f neighborhoodPoint = params.fixedImage.toGlobal(Vec3f(Vec3i(fxi,fyi,fzi))); 

                    Vec3f movingImagePoint = neighborhoodPoint;
                    if (params.useDisplacementField)
                        movingImagePoint += params.displacementField.atGlobal(neighborhoodPoint);

                    ScalarType support;
                    ScalarType sigmaP;
                    Mat3f covP;
                    if (UseWeightImage) {
                        sigmaP  = params.kernel.getSigmaAtPoint(neighborhoodPoint);
                        support = params.kernel.getRegionSupport(sigmaP);
                    } else if (UseWeightTensor) {
                        covP = params.kernel.getCovarianceAtPoint(neighborhoodPoint);
                        support = params.kernel.getRegionSupport(covP);
                    } else {
                        support = params.kernel.getRegionSupport();
                    }

                    Vec3f indexPoint = params.cpImage.toLocal(neighborhoodPoint);
                    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
                    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

                    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
                        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
                            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                                ScalarType k;
                                if (UseWeightImage)
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i, sigmaP);
                                else if (UseWeightTensor)
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i, covP);
                                else
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i);

                                movingImagePoint += params.cpwImage.at(Vec3i(xi, yi, zi))*k*params.cpImage.at(Vec3i(xi, yi, zi));
                            }
                        }
                    }

                    ScalarType kw = neighborhoodWeight(fixedImagePoint, neighborhoodPoint, support, 10);

                    // ScalarType kw;
                    // if (UseWeightImage)
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint, sigmaP);
                    // else if (UseWeightTensor)
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint, covP);
                    // else
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint);

                    Vec3f movingImageGradient(0.0);
                    if (kw>0.0 && params.movingImage.insideGlobal(movingImagePoint)) {
                        ScalarType movingImageValue;
                        params.movingImage.derivsGlobal(movingImagePoint, movingImageValue, movingImageGradient);
                        movingImageValue *= kw;

                        ScalarType fixedImageValue = kw * params.fixedImage.at(Vec3i(fxi,fyi,fzi));

                        sf  += fixedImageValue;
                        sm  += movingImageValue;
                        sff += fixedImageValue  * fixedImageValue;
                        smm += movingImageValue * movingImageValue;
                        sfm += fixedImageValue  * movingImageValue;
                        
                        insideRegion++;
                    }
                }
            }
        }
        //printf("region: %d\n", insideRegion);
        if(insideRegion > 0){
            ScalarType msf = sf/insideRegion;
            ScalarType msm = sm/insideRegion;

            ScalarType d1 = (sff - static_cast<ScalarType>(2.0)*msf*sf + insideRegion*msf*msf);
            ScalarType d2 = (smm - static_cast<ScalarType>(2.0)*msm*sm + insideRegion*msm*msm);

            if(d1>0 && d2>0){
                ccvalue = sfm - msm*sf - msf*sm + insideRegion*msf*msm;
                ccvalue = -ccvalue*ccvalue / (d1 * d2);
            }
        }

    Vec3f movingImageGradient(0.0);
    int inside = 0;
    if (params.movingImage.insideGlobal(centerRegionPoint)) {
        ScalarType movingImageValue;
        params.movingImage.derivsGlobal(centerRegionPoint, movingImageValue, movingImageGradient);
        inside = 1;
    }

    lcc_params.ccvalues [x + y*size[0] + z*size[0]*size[1]] = ccvalue;
    params.gradients    [x + y*size[0] + z*size[0]*size[1]] = movingImageGradient;
    params.pixelsCounted[x + y*size[0] + z*size[0]*size[1]] = inside;
}

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDerivativeLCC(GpuParams params)
{
    Vec3i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

#ifdef REDUCE_REGISTER_SPILLING
    int t_idx = blockDim.x*(threadIdx.x+blockDim.y*(threadIdx.y+blockDim.z*threadIdx.z));
    extern __shared__ Vec3f p_store[];
#endif

    const Vec3f globalPoint = params.cpImage.toGlobal(Vec3f(Vec3i(x, y, z)));

    ScalarType support;
    ScalarType sigmaP;
    Mat3f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    const Vec3f indexPoint = params.fixedImage.toLocal(globalPoint);
    Vec3i imgLower, imgUpper;
    if (DoSubsample) {
        imgLower = std::min(std::max(Vec3i((indexPoint - support/params.fixedImage.scale())/float(params.subsample) + 1.0), Vec3i(0)), params.subsampledSize - 1);
        imgUpper = std::min(std::max(Vec3i((indexPoint + support/params.fixedImage.scale())/float(params.subsample)      ), Vec3i(0)), params.subsampledSize - 1);
    } else {
        imgLower = std::min(std::max(Vec3i(indexPoint - support/params.fixedImage.scale() + 1.0), Vec3i(0)), params.fixedImage.size() - 1);
        imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.fixedImage.scale()      ), Vec3i(0)), params.fixedImage.size() - 1);
    }

    Vec3f derivative(0.0);
    ScalarType u = 0.0;
    ScalarType v = 0.0;
    Vec3f u_dev(0.0);
    Vec3f v_dev(0.0);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {

    // 3x nested for loop is transformed in a flat for loop
    // to optimize for low register spilling
        // const unsigned short diffImg2 = imgUpper[2]-imgLower[2];
        // const unsigned short diffImg1 = imgUpper[1]-imgLower[1];
        // const unsigned short diffImg0 = imgUpper[0]-imgLower[0];
        // for(unsigned iImg=0; iImg<diffImg2*diffImg1*diffImg0; iImg++){{{
        //         int zi = imgLower[2] + iImg / (diffImg0 * diffImg1);
        //         int yi = imgLower[1] + iImg /  diffImg0 % diffImg1;
        //         int xi = imgLower[0] + iImg %  diffImg0;

                int idx;
                if (DoSubsample)
                    idx = xi + params.subsampledSize[0]*(yi + params.subsampledSize[1]*zi);
                else
                    idx = params.fixedImage.toIndex(Vec3i(xi, yi, zi));

                Vec3f point_i;
                if (DoSubsample)
                    point_i = params.fixedImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)*params.subsample + params.gridShift[idx]));
                else
                    point_i = params.fixedImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));

                ScalarType support;
                ScalarType sigmaP;
                Mat3f covP;
                if (UseWeightImage) {
                    sigmaP  = params.kernel.getSigmaAtPoint(point_i);
                    support = params.kernel.getRegionSupport(sigmaP);
                } else if (UseWeightTensor) {
                    covP = params.kernel.getCovarianceAtPoint(point_i);
                    support = params.kernel.getRegionSupport(covP);
                } else {
                    support = params.kernel.getRegionSupport();
                }

                
                // image region arrount point_i
                const Vec3f fIndexPoint = params.fixedImage.toLocal(point_i);
                const Vec3i fImgLower   = std::min(std::max(Vec3i(fIndexPoint - support/params.fixedImage.scale() + 1.0), Vec3i(0)), params.fixedImage.size() - 1);
                const Vec3i fImgUpper   = std::min(std::max(Vec3i(fIndexPoint + support/params.fixedImage.scale()      ), Vec3i(0)), params.fixedImage.size() - 1);

#ifndef NOAPPROX
                
                #ifndef REDUCE_REGISTER_SPILLING
                unsigned archeIdx = 0;

                Vec3f archePoints[NUM_INTERPOLANTS];
                Vec3f archeDisps[NUM_INTERPOLANTS];

                #else
                // use shared memory container for archePoints and archeDisps
                unsigned archeIdx = t_idx + 3; // first indices are reserved for MKCKA, MKCMKCKA and FKCMKA

                Vec3f* archePoints = p_store + archeIdx;
                Vec3f* archeDisps  = p_store + archeIdx + NUM_INTERPOLANTS;

                archeIdx=0;
                #endif

                archePoints[archeIdx++] = point_i; // center point of neighborhood

                #if NUM_INTERPOLANTS >= 9
                Vec3i halfLower = std::min(std::max(Vec3i(fIndexPoint - support/4.0/params.fixedImage.scale() + 1.0), Vec3i(0)), params.fixedImage.size() - 1);
                Vec3i halfUpper = std::min(std::max(Vec3i(fIndexPoint + support/4.0/params.fixedImage.scale()      ), Vec3i(0)), params.fixedImage.size() - 1);

                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(halfLower));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(halfUpper));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfLower[0],halfUpper[1],halfLower[2])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfUpper[0],halfUpper[1],halfLower[2])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfUpper[0],halfLower[1],halfLower[2])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfLower[0],halfLower[1],halfUpper[2])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfUpper[0],halfLower[1],halfUpper[2])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec3f(Vec3i(halfLower[0],halfUpper[1],halfUpper[2])));
                #endif

        
                for(unsigned a=0; a<NUM_INTERPOLANTS; a++){
                    archeDisps[a] = Vec3f(0.0);

                    // control point region
                    const Vec3f cpIndexPoint = params.cpImage.toLocal(point_i);
                    const Vec3i cpImgLower   = std::min(std::max(Vec3i(cpIndexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
                    const Vec3i cpImgUpper   = std::min(std::max(Vec3i(cpIndexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

                    for (int cpzi = cpImgLower[2]; cpzi <= cpImgUpper[2]; ++cpzi) {
                        for (int cpyi = cpImgLower[1]; cpyi <= cpImgUpper[1]; ++cpyi) {
                            for (int cpxi = cpImgLower[0]; cpxi <= cpImgUpper[0]; ++cpxi) {
                                const Vec3f cp_i = params.cpImage.toGlobal(Vec3f(Vec3i(cpxi, cpyi, cpzi)));
                                ScalarType k;
                                if (UseWeightImage)
                                    k = params.kernel.evaluate(cp_i, point_i, sigmaP);
                                else if (UseWeightTensor)
                                    k = params.kernel.evaluate(cp_i, point_i, covP);
                                else
                                    k = params.kernel.evaluate(cp_i, point_i);

                                archeDisps[a] += params.cpwImage.at(Vec3i(cpxi, cpyi, cpzi))*k*params.cpImage.at(Vec3i(cpxi, cpyi, cpzi));
                            }
                        }
                    }
                }
#endif

                ScalarType sf    = 0.0;
                ScalarType sm    = 0.0;
                ScalarType sff   = 0.0;
                ScalarType smm   = 0.0;
                ScalarType sfm   = 0.0;

#ifndef REDUCE_REGISTER_SPILLING
                Vec3f      mkcka(0.0);
                Vec3f      mkcmkcka(0.0);
                Vec3f      fkcmka(0.0);
#else
                #define MKCKA 0
                #define MKCMKCKA 1
                #define FKCMKA 2

                p_store[t_idx+MKCKA]    = Vec3f(0.0);
                p_store[t_idx+MKCMKCKA] = Vec3f(0.0);
                p_store[t_idx+FKCMKA]   = Vec3f(0.0);
#endif
                
                int insideRegion = 0;
                for (int fzi = fImgLower[2]; fzi <= fImgUpper[2]; fzi+=params.subsampleNeighborhood) { 
                    for (int fyi = fImgLower[1]; fyi <= fImgUpper[1]; fyi+=params.subsampleNeighborhood) {      // loop over fixed image kernel region
                        for (int fxi = fImgLower[0]; fxi <= fImgUpper[0]; fxi+=params.subsampleNeighborhood) {

                            Vec3f neighborhoodPoint = params.fixedImage.toGlobal(Vec3f(Vec3i(fxi,fyi,fzi))); 

#ifndef NOAPPROX
                            // approximation: moving point is displaced as the center point of the neighborhood
                            //Vec3f movingImagePoint = neighborhoodPoint + centerDisp;
                            
                            // NN interpolation
                            ScalarType nearest = 1e100;
                            unsigned ni = 0;
                            for(unsigned a=0; a<NUM_INTERPOLANTS; a++){
                                ScalarType l = (archePoints[a]-neighborhoodPoint).length();
                                if(l < nearest){
                                    nearest = l;
                                    ni = a;
                                }
                            }

                            Vec3f movingImagePoint = neighborhoodPoint+archeDisps[ni];
#else
                            Vec3f movingImagePoint = neighborhoodPoint;
#endif

                            if (params.useDisplacementField)
                                movingImagePoint += params.displacementField.atGlobal(neighborhoodPoint);

#ifdef NOAPPROX
                            ScalarType support;
                            ScalarType sigmaP;
                            Mat3f covP;
                            if (UseWeightImage) {
                                sigmaP  = params.kernel.getSigmaAtPoint(neighborhoodPoint);
                                support = params.kernel.getRegionSupport(sigmaP);
                            } else if (UseWeightTensor) {
                                covP = params.kernel.getCovarianceAtPoint(neighborhoodPoint);
                                support = params.kernel.getRegionSupport(covP);
                            } else {
                                support = params.kernel.getRegionSupport();
                            }

                            const Vec3f ncpIndexPoint = params.cpImage.toLocal(neighborhoodPoint);
                            const Vec3i ncpImgLower = std::min(std::max(Vec3i(ncpIndexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
                            const Vec3i ncpImgUpper = std::min(std::max(Vec3i(ncpIndexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

                            for (int nzi = ncpImgLower[2]; nzi <= ncpImgUpper[2]; ++nzi) {
                                for (int nyi = ncpImgLower[1]; nyi <= ncpImgUpper[1]; ++nyi) {
                                    for (int nxi = ncpImgLower[0]; nxi <= ncpImgUpper[0]; ++nxi) {
                                        const Vec3f cp_i = params.cpImage.toGlobal(Vec3f(Vec3i(nxi, nyi, nzi)));
                                        ScalarType k;
                                        if (UseWeightImage)
                                            k = params.kernel.evaluate(cp_i, neighborhoodPoint, sigmaP);
                                        else if (UseWeightTensor)
                                            k = params.kernel.evaluate(cp_i, neighborhoodPoint, covP);
                                        else
                                            k = params.kernel.evaluate(cp_i, neighborhoodPoint);

                                        movingImagePoint += params.cpwImage.at(Vec3i(nxi, nyi, nzi))*k*params.cpImage.at(Vec3i(nxi, nyi, nzi));
                                    }
                                }
                            }
#endif            

                            ScalarType kw = neighborhoodWeight(point_i, neighborhoodPoint, support, 10);

                            ScalarType ka;
                            if (UseWeightImage){
                                //kw  = params.kernel.evaluate(point_i, neighborhoodPoint, sigmaP);
                                ka  = params.kernel.evaluate(globalPoint, neighborhoodPoint, sigmaP);
                            }
                            else if (UseWeightTensor){
                                //kw  = params.kernel.evaluate(point_i, neighborhoodPoint, covP);
                                ka  = params.kernel.evaluate(globalPoint, neighborhoodPoint, covP);
                            }
                            else{
                                //kw  = params.kernel.evaluate(point_i, neighborhoodPoint);
                                ka  = params.kernel.evaluate(globalPoint, neighborhoodPoint);
                            }

                            Vec3f movingImageGradient(0.0);
                            if (kw>0.0 && params.movingImage.insideGlobal(movingImagePoint)) {

                                ScalarType movingImageValue;
                                params.movingImage.derivsGlobal(movingImagePoint, movingImageValue, movingImageGradient);
                                movingImageValue *= kw;

                                ScalarType fixedImageValue = kw * params.fixedImage.at(Vec3i(fxi,fyi,fzi));

                                sf  += fixedImageValue;
                                sm  += movingImageValue;
                                sff += fixedImageValue  * fixedImageValue;
                                smm += movingImageValue * movingImageValue;
                                sfm += fixedImageValue  * movingImageValue;

#ifndef REDUCE_REGISTER_SPILLING
                                mkcmkcka += movingImageValue * (movingImageGradient*kw) * ka;
                                mkcka    += (movingImageGradient*kw) * ka;
                                fkcmka   += fixedImageValue  * (movingImageGradient*kw) * ka;
#else
                                p_store[t_idx+MKCMKCKA] += movingImageValue * (movingImageGradient*kw) * ka;
                                p_store[t_idx+MKCKA]    += (movingImageGradient*kw) * ka;
                                p_store[t_idx+FKCMKA]   += fixedImageValue  * (movingImageGradient*kw) * ka; 
#endif            

                                insideRegion++;
                            }

                        }
                    }
                }

#ifdef EXEC_SINGLE_THREAD
                printf("region: %d",insideRegion);
#endif
                if(insideRegion > 0){

                    ScalarType msf = sf/insideRegion;
                    ScalarType msm = sm/insideRegion;

                    ScalarType d1 = (sff - static_cast<ScalarType>(2.0)*msf*sf + insideRegion*msf*msf);

                    u = d1 * (smm - static_cast<ScalarType>(2.0)*msm*sm + insideRegion*msm*msm);
                    v = sfm - msm*sf - msf*sm + insideRegion*msf*msm;

#ifndef REDUCE_REGISTER_SPILLING
                    v_dev = static_cast<ScalarType>(2.0)*v  * (fkcmka   - msf*mkcka);
                    u_dev = static_cast<ScalarType>(2.0)*d1 * (mkcmkcka - msm*mkcka);
#else
                    v_dev = static_cast<ScalarType>(2.0)*v  * (p_store[t_idx+FKCMKA]   - msf*p_store[t_idx+MKCKA]);
                    u_dev = static_cast<ScalarType>(2.0)*d1 * (p_store[t_idx+MKCMKCKA] - msm*p_store[t_idx+MKCKA]);
#endif

                    v = v*v;

                    if(std::abs(u)>1e-16)
                        derivative += -(v_dev*u - u_dev*v) / (u*u);
                    // TODO: what if u==0?
                }
            }
        }
    }

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec3i(x, y, z));
    for (unsigned d = 0; d < SpaceDimensions; d++){
        params.derivatives[pixelIndex + d*dimensionStride] = derivative[d];
    }
}

#else

//////////////  2D /////////////////

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateValueLCC(GpuParams params, GpuLCCParams lcc_params)
{
    Vec2i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2i samplePos = Vec2i(x, y);
    if (DoSubsample)
        samplePos = samplePos*params.subsample + params.gridShift[x + y*size[0]];
    Vec2f  fixedImagePoint = params.fixedImage.toGlobal(Vec2f(samplePos));
    Vec2f centerRegionPoint = fixedImagePoint;
    if (params.useDisplacementField)
        centerRegionPoint += params.displacementField.atGlobal(fixedImagePoint);

    ScalarType support;
    ScalarType sigmaP;
    Mat2f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec2f indexPoint = params.cpImage.toLocal(fixedImagePoint);
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);


        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec2f point_i = params.cpImage.toGlobal(Vec2f(Vec2i(xi, yi)));
                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, covP);
                else
                    k = params.kernel.evaluate(fixedImagePoint, point_i);

                centerRegionPoint += params.cpwImage.at(Vec2i(xi, yi))*k*params.cpImage.at(Vec2i(xi, yi));
            }
        }
    
    // TODO: make subsampled fixed image region as in gather gradients
        Vec2f fIndexPoint = params.fixedImage.toLocal(fixedImagePoint);
        Vec2i fImgLower = std::min(std::max(Vec2i(fIndexPoint - support/params.fixedImage.scale() + 1.0), Vec2i(0)), params.fixedImage.size() - 1);
        Vec2i fImgUpper = std::min(std::max(Vec2i(fIndexPoint + support/params.fixedImage.scale()      ), Vec2i(0)), params.fixedImage.size() - 1);

        ScalarType sf    = 0.0;
        ScalarType sm    = 0.0;
        ScalarType sff   = 0.0;
        ScalarType smm   = 0.0;
        ScalarType sfm   = 0.0;
        int insideRegion = 0;
            for (int fyi = fImgLower[1]; fyi <= fImgUpper[1]; fyi+=params.subsampleNeighborhood) {      // loop over fixed image kernel region
                for (int fxi = fImgLower[0]; fxi <= fImgUpper[0]; fxi+=params.subsampleNeighborhood) {
                    // TODO: do subsampling as in gatherGradients
                    Vec2f neighborhoodPoint = params.fixedImage.toGlobal(Vec2f(Vec2i(fxi,fyi))); 

                    Vec2f movingImagePoint = neighborhoodPoint;
                    if (params.useDisplacementField)
                        movingImagePoint += params.displacementField.atGlobal(neighborhoodPoint);

                    ScalarType support;
                    ScalarType sigmaP;
                    Mat2f covP;
                    if (UseWeightImage) {
                        sigmaP  = params.kernel.getSigmaAtPoint(neighborhoodPoint);
                        support = params.kernel.getRegionSupport(sigmaP);
                    } else if (UseWeightTensor) {
                        covP = params.kernel.getCovarianceAtPoint(neighborhoodPoint);
                        support = params.kernel.getRegionSupport(covP);
                    } else {
                        support = params.kernel.getRegionSupport();
                    }

                    Vec2f indexPoint = params.cpImage.toLocal(neighborhoodPoint);
                    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
                    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

                        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
                            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                                Vec2f point_i = params.cpImage.toGlobal(Vec2f(Vec2i(xi, yi)));
                                ScalarType k;
                                if (UseWeightImage)
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i, sigmaP);
                                else if (UseWeightTensor)
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i, covP);
                                else
                                    k = params.kernel.evaluate(neighborhoodPoint, point_i);

                                movingImagePoint += params.cpwImage.at(Vec2i(xi, yi))*k*params.cpImage.at(Vec2i(xi, yi));
                            }
                        }

                    ScalarType kw = neighborhoodWeight(fixedImagePoint, neighborhoodPoint, support, 10);

                    // ScalarType kw;
                    // if (UseWeightImage)
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint, sigmaP);
                    // else if (UseWeightTensor)
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint, covP);
                    // else
                    //     kw = params.kernel.evaluate(fixedImagePoint, neighborhoodPoint);

                    Vec2f movingImageGradient(0.0);
                    if (kw>0.0 && params.movingImage.insideGlobal(movingImagePoint)) {
                        ScalarType movingImageValue;
                        params.movingImage.derivsGlobal(movingImagePoint, movingImageValue, movingImageGradient);
                        movingImageValue *= kw;

                        ScalarType fixedImageValue = kw * params.fixedImage.at(Vec2i(fxi,fyi));

                        sf  += fixedImageValue;
                        sm  += movingImageValue;
                        sff += fixedImageValue  * fixedImageValue;
                        smm += movingImageValue * movingImageValue;
                        sfm += fixedImageValue  * movingImageValue;
                        
                        insideRegion++;
                    }
                }
            }
        
        //printf("region: %d\n", insideRegion);
        ScalarType ccvalue=0.0;
        if(insideRegion > 0){
            ScalarType msf = sf/insideRegion;
            ScalarType msm = sm/insideRegion;

            ScalarType d1 = (sff - static_cast<ScalarType>(2.0)*msf*sf + insideRegion*msf*msf);
            ScalarType d2 = (smm - static_cast<ScalarType>(2.0)*msm*sm + insideRegion*msm*msm);

            if(d1>0 && d2>0){
                ccvalue = sfm - msm*sf - msf*sm + insideRegion*msf*msm;
                ccvalue = -ccvalue*ccvalue / (d1 * d2);
            }
        }

    Vec2f movingImageGradient(0.0);
    int inside = 0;
    if (params.movingImage.insideGlobal(centerRegionPoint)) {
        ScalarType movingImageValue;
        params.movingImage.derivsGlobal(centerRegionPoint, movingImageValue, movingImageGradient);
        inside = 1;
    }

    lcc_params.ccvalues [x + y*size[0]] = ccvalue;
    params.gradients    [x + y*size[0]] = movingImageGradient;
    params.pixelsCounted[x + y*size[0]] = inside;
}

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDerivativeLCC(GpuParams params)
{
    Vec2i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;
    
    Vec2f globalPoint = params.cpImage.toGlobal(Vec2f(Vec2i(x, y)));
    ScalarType w_i = params.cpwImage.at(Vec2i(x, y));

    ScalarType support;
    ScalarType sigmaP;
    Mat2f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec2f indexPoint = params.fixedImage.toLocal(globalPoint);
    Vec2i imgLower, imgUpper;
    if (DoSubsample) {
        imgLower = std::min(std::max(Vec2i((indexPoint - support/params.fixedImage.scale())/float(params.subsample) + 1.0), Vec2i(0)), params.subsampledSize - 1);
        imgUpper = std::min(std::max(Vec2i((indexPoint + support/params.fixedImage.scale())/float(params.subsample)      ), Vec2i(0)), params.subsampledSize - 1);
    } else {
        imgLower = std::min(std::max(Vec2i(indexPoint - support/params.fixedImage.scale() + 1.0), Vec2i(0)), params.fixedImage.size() - 1);
        imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.fixedImage.scale()      ), Vec2i(0)), params.fixedImage.size() - 1);
    }

    Vec2f derivative(0.0);
    ScalarType u = 0.0;
    ScalarType v = 0.0;
    Vec2f u_dev(0.0);
    Vec2f v_dev(0.0);
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {

                int idx;
                if (DoSubsample)
                    idx = xi + params.subsampledSize[0]*yi;
                else
                    idx = params.fixedImage.toIndex(Vec2i(xi, yi));

                Vec2f point_i;
                if (DoSubsample)
                    point_i = params.fixedImage.toGlobal(Vec2f(Vec2i(xi, yi)*params.subsample + params.gridShift[idx]));
                else
                    point_i = params.fixedImage.toGlobal(Vec2f(Vec2i(xi, yi)));


                ScalarType support;
                ScalarType sigmaP;
                Mat2f covP;
                if (UseWeightImage) {
                    sigmaP  = params.kernel.getSigmaAtPoint(point_i);
                    support = params.kernel.getRegionSupport(sigmaP);
                } else if (UseWeightTensor) {
                    covP = params.kernel.getCovarianceAtPoint(point_i);
                    support = params.kernel.getRegionSupport(covP);
                } else {
                    support = params.kernel.getRegionSupport();
                }

                
                // image region arrount point_i
                Vec2f fIndexPoint = params.fixedImage.toLocal(point_i);
                Vec2i fImgLower = std::min(std::max(Vec2i(fIndexPoint - support/params.fixedImage.scale() + 1.0), Vec2i(0)), params.fixedImage.size() - 1);
                Vec2i fImgUpper = std::min(std::max(Vec2i(fIndexPoint + support/params.fixedImage.scale()      ), Vec2i(0)), params.fixedImage.size() - 1);

#ifndef NOAPPROX
                // store points for nearest neighbor interpolation
                Vec2f archePoints[NUM_INTERPOLANTS];
                Vec2f archeDisps[NUM_INTERPOLANTS];

                unsigned archeIdx = 0;

                archePoints[archeIdx++] = point_i; // center point of neighborhood

            #if NUM_INTERPOLANTS >= 5 // corner points of neighborhood
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(fImgLower));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(fImgUpper));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(Vec2i(fImgLower[0],fImgUpper[1])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(Vec2i(fImgUpper[0],fImgLower[1])));
            #endif
            #if NUM_INTERPOLANTS >=9 // half distance corner point of neighborhood
                Vec2i halfLower = std::min(std::max(Vec2i(fIndexPoint - support/3.0/params.fixedImage.scale() + 1.0), Vec2i(0)), params.fixedImage.size() - 1);
                Vec2i halfUpper = std::min(std::max(Vec2i(fIndexPoint + support/3.0/params.fixedImage.scale()      ), Vec2i(0)), params.fixedImage.size() - 1);

                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(halfLower));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(halfUpper));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(Vec2i(halfLower[0],halfUpper[1])));
                archePoints[archeIdx++] = params.fixedImage.toGlobal(Vec2f(Vec2i(halfUpper[0],halfLower[1])));
            #endif


                for(unsigned a=0; a<NUM_INTERPOLANTS; a++){
                    archeDisps[a] = Vec2f(0.0);

                    // control point region
                    Vec2f cpIndexPoint = params.cpImage.toLocal(archePoints[a]);
                    Vec2i cpImgLower = std::min(std::max(Vec2i(cpIndexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
                    Vec2i cpImgUpper = std::min(std::max(Vec2i(cpIndexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

                    for (int cpyi = cpImgLower[1]; cpyi <= cpImgUpper[1]; ++cpyi) {
                        for (int cpxi = cpImgLower[0]; cpxi <= cpImgUpper[0]; ++cpxi) {
                            Vec2f cp_i = params.cpImage.toGlobal(Vec2f(Vec2i(cpxi, cpyi)));
                            ScalarType k;
                            if (UseWeightImage)
                                k = params.kernel.evaluate(cp_i, archePoints[a], sigmaP);
                            else if (UseWeightTensor)
                                k = params.kernel.evaluate(cp_i, archePoints[a], covP);
                            else
                                k = params.kernel.evaluate(cp_i, archePoints[a]);

                            archeDisps[a] += params.cpwImage.at(Vec2i(cpxi, cpyi))*k*params.cpImage.at(Vec2i(cpxi, cpyi));
                        }
                    }
                }
#endif

                ScalarType sf    = 0.0;
                ScalarType sm    = 0.0;
                ScalarType sff   = 0.0;
                ScalarType smm   = 0.0;
                ScalarType sfm   = 0.0;
                Vec2f      mkcka(0.0);
                Vec2f      mkcmkcka(0.0);
                Vec2f      fkcmka(0.0);
                int insideRegion = 0;

                    for (int fyi = fImgLower[1]; fyi <= fImgUpper[1]; fyi+=params.subsampleNeighborhood) {      // loop over fixed image kernel region
                        for (int fxi = fImgLower[0]; fxi <= fImgUpper[0]; fxi+=params.subsampleNeighborhood) {

                            Vec2f neighborhoodPoint = params.fixedImage.toGlobal(Vec2f(Vec2i(fxi, fyi)));

#ifndef NOAPPROX
                            // NN interpolation
                            ScalarType nearest = 1e100;
                            unsigned ni = 0;
                            for(unsigned a=0; a<NUM_INTERPOLANTS; a++){
                                ScalarType l = (archePoints[a]-neighborhoodPoint).length();
                                if(l < nearest){
                                    nearest = l;
                                    ni = a;
                                }
                            }

                            Vec2f movingImagePoint = neighborhoodPoint+archeDisps[ni];
#else
                            Vec2f movingImagePoint = neighborhoodPoint;
#endif

                            if (params.useDisplacementField)
                                movingImagePoint += params.displacementField.atGlobal(neighborhoodPoint);

#ifdef NOAPPROX
                            ScalarType support;
                            ScalarType sigmaP;
                            Mat2f covP;
                            if (UseWeightImage) {
                                sigmaP  = params.kernel.getSigmaAtPoint(neighborhoodPoint);
                                support = params.kernel.getRegionSupport(sigmaP);
                            } else if (UseWeightTensor) {
                                covP = params.kernel.getCovarianceAtPoint(neighborhoodPoint);
                                support = params.kernel.getRegionSupport(covP);
                            } else {
                                support = params.kernel.getRegionSupport();
                            }

                            Vec2f ncpIndexPoint = params.cpImage.toLocal(neighborhoodPoint);
                            Vec2i ncpImgLower = std::min(std::max(Vec2i(ncpIndexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
                            Vec2i ncpImgUpper = std::min(std::max(Vec2i(ncpIndexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

                                for (int nyi = ncpImgLower[1]; nyi <= ncpImgUpper[1]; ++nyi) {
                                    for (int nxi = ncpImgLower[0]; nxi <= ncpImgUpper[0]; ++nxi) {
                                        Vec2f cp_i = params.cpImage.toGlobal(Vec2f(Vec2i(nxi, nyi)));
                                        ScalarType k;
                                        if (UseWeightImage)
                                            k = params.kernel.evaluate(neighborhoodPoint, cp_i, sigmaP);
                                        else if (UseWeightTensor)
                                            k = params.kernel.evaluate(neighborhoodPoint, cp_i, covP);
                                        else
                                            k = params.kernel.evaluate(neighborhoodPoint, cp_i);

                                        movingImagePoint += params.cpwImage.at(Vec2i(nxi, nyi))*k*params.cpImage.at(Vec2i(nxi, nyi));
                                    }
                                }
#endif            

                            // neighborhood weight value
                            ScalarType kw = neighborhoodWeight(point_i, neighborhoodPoint, support, 10); 

                            ScalarType ka;
                            if (UseWeightImage){
                                //kw = params.kernel.evaluate(point_i, neighborhoodPoint, sigmaP);
                                ka = params.kernel.evaluate(globalPoint, neighborhoodPoint, sigmaP);
                            }
                            else if (UseWeightTensor){
                                //kw = params.kernel.evaluate(point_i, neighborhoodPoint, covP);
                                ka = params.kernel.evaluate(globalPoint, neighborhoodPoint, covP);
                            }
                            else{
                                //kw = params.kernel.evaluate(point_i, neighborhoodPoint);
                                ka = params.kernel.evaluate(globalPoint, neighborhoodPoint);
                            }

                            Vec2f movingImageGradient(0.0);
                            if (kw>0.0 && params.movingImage.insideGlobal(movingImagePoint)) {
                                // calculate lcc derivative

                                ScalarType movingImageValue;
                                params.movingImage.derivsGlobal(movingImagePoint, movingImageValue, movingImageGradient);
                                movingImageValue *= kw;

                                ScalarType fixedImageValue = kw * params.fixedImage.at(Vec2i(fxi,fyi));

                                sf  += fixedImageValue;
                                sm  += movingImageValue;
                                sff += fixedImageValue  * fixedImageValue;
                                smm += movingImageValue * movingImageValue;
                                sfm += fixedImageValue  * movingImageValue;
                                
                                mkcmkcka += movingImageValue * (movingImageGradient*kw) * ka;
                                mkcka    += (movingImageGradient*kw) * ka;
                                fkcmka   += fixedImageValue *  (movingImageGradient*kw) * ka;

                                insideRegion++;
                            }

                        }
                    }

                //printf("region: %d",insideRegion);
                if(insideRegion > 0){
                    ScalarType msf = sf/insideRegion;
                    ScalarType msm = sm/insideRegion;

                    ScalarType d1 = (sff - static_cast<ScalarType>(2.0)*msf*sf + insideRegion*msf*msf);

                    u = d1 * (smm - static_cast<ScalarType>(2.0)*msm*sm + insideRegion*msm*msm);
                    v = sfm - msm*sf - msf*sm + insideRegion*msf*msm;

                    v_dev = static_cast<ScalarType>(2.0)*v  * (fkcmka   - msf*mkcka);
                    u_dev = static_cast<ScalarType>(2.0)*d1 * (mkcmkcka - msm*mkcka);                    
                    
                    v = v*v;

                    if(std::abs(u)>1e-16)
                        derivative += -(v_dev*u - u_dev*v) / (u*u);
                }
            }
        }

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec2i(x, y));
    for (unsigned d = 0; d < SpaceDimensions; d++){
        params.derivatives[pixelIndex + d*dimensionStride] = derivative[d];
    }
}

#endif // SpaceDimensions == 3


template<bool DoSubsample>
void resolveValueLCC(dim3 gridDim, dim3 blockDim, GpuParams params, GpuLCCParams lcc_params)
{
        if (params.kernel.useWeightImage())
            evaluateValueLCC<DoSubsample, true,  false><<<gridDim, blockDim>>>(params, lcc_params);
        else if (params.kernel.useWeightTensor())
            evaluateValueLCC<DoSubsample, false, true> <<<gridDim, blockDim>>>(params, lcc_params);
        else
            evaluateValueLCC<DoSubsample, false, false><<<gridDim, blockDim>>>(params, lcc_params);
}

template<bool DoSubsample>
void resolveDerivativeLCC(dim3 gridDim, dim3 blockDim, GpuParams params)
{

    #if !defined REDUCE_REGISTER_SPILLING || SpaceDimensions == 2
    if (params.kernel.useWeightImage())
        evaluateDerivativeLCC<DoSubsample, true, false><<<gridDim, blockDim>>>(params);
    else if (params.kernel.useWeightTensor())
        evaluateDerivativeLCC<DoSubsample, false, true><<<gridDim, blockDim>>>(params);
    else
        evaluateDerivativeLCC<DoSubsample, false, false><<<gridDim, blockDim>>>(params);
    #else
    // calculate size of shared memory to hold few float vectors
    unsigned p_store_size = 3 + 2*NUM_INTERPOLANTS;
    unsigned store_size = p_store_size*blockDim.x*blockDim.y*blockDim.z*sizeof(Vec3f);

    if (params.kernel.useWeightImage())
        evaluateDerivativeLCC<DoSubsample, true, false><<<gridDim, blockDim, store_size>>>(params);
    else if (params.kernel.useWeightTensor())
        evaluateDerivativeLCC<DoSubsample, false, true><<<gridDim, blockDim, store_size>>>(params);
    else
        evaluateDerivativeLCC<DoSubsample, false, false><<<gridDim, blockDim, store_size>>>(params);
    #endif
}