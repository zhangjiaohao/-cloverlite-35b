#include "preview_conv.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 1; } } while(0)

int main() {
    constexpr int C = 8192;
    float worst = 0;
    for (int T : {1, 2, 4, 8, 16, 32}) {
        std::vector<float> x(T*C), hist(C*3), w(C*4), right(C*3), result(T*C), after(C*3);
        for (size_t i=0; i<x.size(); ++i) x[i] = std::sin(float(i)*.017f);
        for (size_t i=0; i<hist.size(); ++i) hist[i] = std::cos(float(i)*.011f);
        for (size_t i=0; i<w.size(); ++i) w[i] = .4f*std::sin(float(i)*.007f);
        for (size_t i=0; i<right.size(); ++i) right[i] = .3f*std::cos(float(i)*.013f);
        float *dx,*dh,*dw,*dr,*dy;
        CHECK(cudaMalloc(&dx,x.size()*4)); CHECK(cudaMalloc(&dh,hist.size()*4));
        CHECK(cudaMalloc(&dw,w.size()*4)); CHECK(cudaMalloc(&dr,right.size()*4)); CHECK(cudaMalloc(&dy,result.size()*4));
        CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dh,hist.data(),hist.size()*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dr,right.data(),right.size()*4,cudaMemcpyHostToDevice));
        clover_diffusion::preview_conv_silu<<<(T*C+255)/256,256>>>(dx,dh,dw,dr,dy,C,T);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(result.data(),dy,result.size()*4,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(after.data(),dh,after.size()*4,cudaMemcpyDeviceToHost));
        if (hist != after) { std::fprintf(stderr,"history modified at width %d\n",T); return 1; }
        float maxerr=0;
        for(int t=0;t<T;++t) for(int c=0;c<C;++c) {
            double sum=0;
            for(int j=0;j<4;++j) {int k=t+j-3;sum+=double(k<0?hist[c*3+k+3]:x[k*C+c])*w[c*4+j];}
            for(int d=1;d<=3;++d) if(t+d<T) sum+=double(x[(t+d)*C+c])*right[c*3+d-1];
            const double expect=sum/(1.+std::exp(-sum));
            maxerr=std::max(maxerr,float(std::abs(result[t*C+c]-expect)));
        }
        std::printf("width %d, channels %d, max error %.8g, prefix unchanged\n",T,C,maxerr);
        worst=std::max(worst,maxerr);
        CHECK(cudaFree(dx)); CHECK(cudaFree(dh)); CHECK(cudaFree(dw)); CHECK(cudaFree(dr)); CHECK(cudaFree(dy));
        if(maxerr>2e-6f) return 1;
    }
    std::printf("PASS: max error %.8g\n",worst);
    return 0;
}
