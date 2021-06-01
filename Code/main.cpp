#include "finalCuda.h"
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define CV_8U   0
#define CV_BGR2GRAY 6 //Constant for color conversion


int main(int argc, char const *argv[]) {
  // Start capturing the images/video
  cv::VideoCapture capture(0);
  cv::Mat          frame; //For storing the actual image in a vector
  //Validating error while trying to capture video
  if(!capture.isOpened()){
    return -1;
  }
  //Fixing sizes for the frame of the video captured
  capture.set(CV_CAP_PROP_FRAME_WIDTH,640); //Video frame's width: 640pix
  capture.set(CV_CAP_PROP_FRAME_HEIGHT,480); //Video frame's height: 480pix

  // CPU captured data
  capture >> frame; //Capturing original image


  //Buffers for the images //CV_8U- 8-bit array- 0 channels
  //Initial image- source image ---- Matrix Original
  cv::Mat original (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
  //Buffer for the results by applying Box Filter- kind of blurr to erase some noise from the Sobel calcs
  cv::Mat blurr  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
  //Buffer for the results by applying Sobel filter- edges
  cv::Mat edge   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
  // SOBEL GRADIENTS
  //GX
  cv::Mat gradX  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
  //GY
  cv::Mat gradY  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));

  //Color conversion
  cv::cvtColor(frame, blurr, CV_BGR2GRAY);
  cv::cvtColor(frame, edge, CV_BGR2GRAY);
  cv::cvtColor(frame, gradX, CV_BGR2GRAY);
  cv::cvtColor(frame, gradY, CV_BGR2GRAY);

  // This are the windows in which the results and captures are going to be seen
  cv::namedWindow("Original"); //Original image

  //Keep capturing the images until break by 27=ESC -- closing window
  while(1){
    capture >> frame; //Capturing image

    cv::cvtColor(frame, original, CV_BGR2GRAY); //color conversion to grayscale

    //Applying the filters
    // Blurr- sobel noise eraser
    boxFilter_host(original.data, blurr.data, frame.size().width,frame.size().height, 3, 3);
    //edge detection
    sobelFilter_host(blurr.data, edge.data,gradX.data, gradY.data, frame.size().width,frame.size().height);

    //Displaying the originals and the resulting atrix

    cv::imshow("Original",frame); //the actual capturing -- Original
    cv::imshow("Edge Detection - Sobel Filter", edge); //Results of sobel filter

    cv::imshow("Edge Detection - Sobel X Filter", gradX); //Results of sobel filter
    cv::imshow("Edge Detection - Sobel Y Filter", gradY); //Results of sobel filter
    cv::imshow("Smoothing image - Blurr Filter", blurr); //Results of sobel filter
    //BORRAR displaying del grayscale
    cv::imshow("Color to Gray (Original)",original); // Original after color conversion - just to check

    // Validando el cierre de la ventana para romper el while(1)
        if(cv::waitKey(1) == 27) break; //27=ESC
  }

  //Freeing memory
  destroyImageBuffer(original.data);
  destroyImageBuffer(blurr.data);
  destroyImageBuffer(edge.data);
  destroyImageBuffer(gradX.data);
  destroyImageBuffer(gradY.data);

  return 0;
}
