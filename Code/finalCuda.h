#ifndef _finalCuda_h
#define _finalCuda_h

typedef unsigned int uint;
typedef unsigned char uchar;

//Main functions headers
void boxFilter_host(uchar *h_original, uchar *h_result, int max_width, int max_heigth, int kernel_w, int kernel_h); // Blurr- sobel noise eraser
void sobelFilter_host(uchar *h_original, uchar *h_result, uchar *gradX, uchar *gradY, int max_width, int max_heigth); //edge detection

unsigned char* createImageBuffer  (uint  bytes);
void   destroyImageBuffer (uchar *bytes);

#endif
