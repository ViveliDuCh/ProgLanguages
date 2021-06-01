#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "finalCuda.h"


// Sobel gradient kernels: This is to see how the kernels are structured for each gradient
/*__constant__  float sobelGradientX[9] =
{
    -1.f, 0.f, 1.f,
    -2.f, 0.f, 2.f,
    -1.f, 0.f, 1.f,
};
__constant__  float sobelGradientY[9] =
{
    1.f, 2.f, 1.f,
    0.f, 0.f, 0.f,
    -1.f, -2.f, -1.f,
};*/

//Applying Sobel gradient kernels to the image and at last applying pythagoran theorem for erasing noise
__global__ void sobelFilter_device (unsigned char *d_original, unsigned char *d_result,unsigned char* d_gradX, unsigned char* d_gradY, int max_width, int max_height){
  int dx, dy;
  //Pixel location
  int col = threadIdx.x + blockIdx.x * blockDim.x;//x
	int row = threadIdx.y + blockIdx.y * blockDim.y;//y

  //Verifying threads are not exceeding the array's boundaries
  if((col > 0) && (col < max_width - 1) && (row > 0) && (row < max_height - 1)){

    //Convolve image with mask

    //Gradient X
    dx= -1*d_original[max_width*(row-1)+(col-1)]   +
        -2*d_original[max_width*(row)+(col-1)]     +
        -1*d_original[max_width*(row+1)+(col-1)]   +
         1*d_original[max_width*(row-1)+(col+1)]   +
         2*d_original[max_width*(row)+(col+1)]     +
         1*d_original[max_width*(row+1)+(col+1)];
        ;
    //Gradient Y
    dy= -1*d_original[max_width*(row-1)+(col-1)]   +
        -2*d_original[max_width*(row-1)+(col)]     +
        -1*d_original[max_width*(row-1)+(col+1)]   +
         1*d_original[max_width*(row+1)+(col-1)]   +
         2*d_original[max_width*(row+1)+(col)]     +
         1*d_original[max_width*(row+1)+(col+1)];
        ;
    // X and Y grad
    d_gradX[(max_width * row) + col] = dx;
    d_gradY[(max_width * row) + col] = dy;
    //the pythagoran theorem: sqrt(dx^2 +dy^2)
    d_result[(max_width*row)+col]= (int) sqrt( (((float)dx)*((float)dx)) + (((float)dy)*((float)dy)) );
  }

}

//Box filter -- simple blurr filter
//3 by 3 sumation average to blurr the image before applying Sobel so we can erase some image noise
__global__ void boxFilter_device (unsigned char *d_original, unsigned char *d_result, int max_width, int max_height, int kernel_w, int kernel_h){

  //This may be __shared__ type of var, initialized to 0 by just the first thread and used later by all
  int count=0;
  float sum=0.0;

  //Pixel location
  int col = threadIdx.x + blockIdx.x * blockDim.x;//x
	int row = threadIdx.y + blockIdx.y * blockDim.y;//y

  //Make the sumation of the surrounding pixels
  for(int k_row = -1*(kernel_h/2);  k_row < (kernel_h/2)+1; k_row ++){
		for(int k_col = -1*(kernel_w/2);  k_col < (kernel_w/2)+1 ; k_col ++){

      //Verifying offset is within image boundaries
      if( ((col+k_col) >= 0 && (col+k_col) < max_width) && ((row+k_row) >= 0 && (row+k_row) < max_height)){
          //Getting the sum of each number
          sum+= (float) d_original[((row+k_row)*max_width) + (col+k_col)];
          //Getting the total amount of numbers
          count++;
      }
		}
	}
  //Finally, calculating the average sum of the actual pixel's neighbours
  sum = sum/((float)count);

  //Putting the result os calcs in the actual pix
  d_result[(max_width*row)+col] = (unsigned char) sum;

}
// Blurr filter = Sobel filter's noise eraser -- Host calling Device function
void boxFilter_host(uchar *h_original, uchar *h_result, int max_width, int max_height, int kernel_w, int kernel_h){
  unsigned char *d_original, *d_result;
  cudaMalloc((void**)&d_original,sizeof(unsigned char)*max_width*max_height);
  cudaMalloc((void**)&d_result,sizeof(unsigned char)*max_width*max_height);

  //Copying CPU original image's data to GPU
  cudaMemcpy(d_original, h_original,sizeof(unsigned char)*max_width*max_height, cudaMemcpyHostToDevice);

  dim3 Blocks(max_width/16,max_height/16);
  dim3 Threads(16,16); //16 threads per block

  boxFilter_device<<< Blocks , Threads >>>(d_original,d_result,max_width,max_height,3,3);

  //Host waiting for the device to finish calculations
  cudaThreadSynchronize();

  //Copying GPU render data back to CPU
	cudaMemcpy(h_result, d_result,sizeof(unsigned char)*max_width*max_height, cudaMemcpyDeviceToHost);
}

// Edge detection -- Host calling Device function
void sobelFilter_host(uchar *h_original, uchar *h_result2, uchar* h_gradX, uchar* h_gradY, int max_width, int max_height){
  unsigned char *d_original, *d_result2, *d_gradX, *d_gradY;

  //Alllocating memmory in GPU
  cudaMalloc((void**)&d_original,sizeof(unsigned char)*max_width*max_height);
  cudaMalloc((void**)&d_result2,sizeof(unsigned char)*max_width*max_height);
  cudaMalloc((void**)&d_gradX, sizeof(unsigned char) * max_width * max_height);
  cudaMalloc((void**)&d_gradY, sizeof(unsigned char) * max_width * max_height);

  //Copying CPU original image's data to GPU
  cudaMemcpy(d_original, h_original,sizeof(unsigned char)*max_width*max_height, cudaMemcpyHostToDevice);

  dim3 Blocks(max_width/16,max_height/16);
  dim3 Threads(16,16); //16 threads per block

  sobelFilter_device<<< Blocks , Threads >>>(d_original,d_result2,d_gradX,d_gradY,max_width,max_height);

  //Host waiting for the device to finish calculations
  cudaThreadSynchronize();

  //Copying GPU render data back to CPU
	cudaMemcpy(h_result2, d_result2,sizeof(unsigned char)* max_width * max_height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradX,   d_gradX, sizeof(unsigned char) * max_width * max_height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradY,   d_gradY, sizeof(unsigned char) * max_width * max_height, cudaMemcpyDeviceToHost);
}

// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(uint bytes){
    unsigned char *ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    return ptr;
}

void   destroyImageBuffer (uchar *bytes){
  cudaFreeHost(bytes);
}
