import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from  pycuda.gpuarray import GPUArray
import numpy as np
import time 
maxStaN=20
nptype=np.float32
mod= SourceModule("""
__global__ void corr( float * a,float* b,  int *laL,int *lbL,float *tb0L,float* c,float *m,float *s,float *mb,float *sb) {
    int la=laL[0],lb=lbL[0];
    float tb0=tb0L[0];
    int i=threadIdx.x+blockIdx.x*blockDim.x,j=0;
    int tid=threadIdx.x;
    float ta=0,tc=0;
    mb=mb+blockIdx.x*blockDim.x;
    sb=sb+blockIdx.x*blockDim.x;
    mb[tid]=0;sb[tid]=0;
    while (i < la-lb+1) {
        ta=0;tc=0;
        for(j=0;j<lb;j++){
            ta+=a[i+j]*a[i+j];
            tc+=a[i+j]*b[j];
        }
        
        *(c+i)= fdividef(tc,sqrtf(ta)*tb0);
        if(ta==0){
           *(c+i)=0;
        }
        mb[tid]+=c[i];
        sb[tid]+=c[i]*c[i];
        i+=blockDim.x*gridDim.x;
    }
   __syncthreads();
   i=blockDim.x/2;
   while(i!=0){
     if(tid<i){
       mb[tid]+=mb[tid+i];
       sb[tid]+=sb[tid+i];
     }
     __syncthreads();
     i/=2;
     }    
      __syncthreads();
    if(tid==0){
     m[blockIdx.x]=mb[0];
     s[blockIdx.x]=sb[0];
     }
    return ;
}

""")

corr = mod.get_function("corr")
def corrPy(a, b, tN=64,bN=128):
    #if not a.dtype==nptype:
    time1=time.time()
    a=a.astype(nptype)
    #if not b.dtype==nptype:
    b=b.astype(nptype)
    aI=drv.In(a)
    bI=drv.In(b)
    la=np.array(a.size,dtype=np.int64)
    lb=np.array(b.size,dtype=np.int64)
    #print(lb)
    c=np.zeros(la-lb+1,nptype)
    cO=drv.InOut(c)
    tb=np.array(np.sqrt((b*b).sum()).astype(nptype),dtype=nptype)
    d_m=np.zeros(bN,dtype=nptype)
    d_s=np.zeros(bN,dtype=nptype)
    b_m=np.zeros(bN*tN,dtype=nptype)
    b_s=np.zeros(bN*tN,dtype=nptype)
    d_mO=drv.InOut(d_m)
    d_sO=drv.InOut(d_s)
    b_mO=drv.InOut(b_m)
    b_sO=drv.InOut(b_s)
    if tb==0:
        m=0
        s=1
    else:
        time2=time.time()
        corr(aI, bI, drv.In(la), drv.In(lb),drv.In(tb),cO,\
            d_mO,d_sO,b_mO,b_sO,block=(tN,1,1),grid=(bN,1))
        time3=time.time()
        m=(d_m/(la-lb+1)).sum()
        s=(d_s/(la-lb+1)).sum()
        s=np.sqrt(np.abs(s-m*m))
    time4=time.time()
    print(time2-time1,time3-time2,time4-time3)
    return c,m,s
