# MultiPoseCV
OpenCV를 사용하여 인공지능으로 인간의 포즈를 인식하는 프로그램입니다.<br/>
살행을 하시면, 프로그램이 컴퓨터에 연결되어 있는 카메라로 인식을 합니다.<br/>
이번에 이 라이브러리를 처음으로 써봐서 잘 모르는 것도 많았지만 완성을 했고 여기다 올리겠습니다.


This is a pose detection program based on C++ using OpenCV library.

# Here's simple steps to run the program..
```
1. Download OpenCV library
2. Add OpenCV library path to windows enviroment variable
3. Link the OpenCV library in Visual Studio settings in the file
4. Add caffe model to coco path
```

### Link to download OpenCV library
https://opencv.org/releases/

# Links explaining the process 1-3 above
https://webnautes.tistory.com/1132<br/>
https://docs.opencv.org/4.x/dd/d6e/tutorial_windows_visual_studio_opencv.html<br/>
https://heisanbug.tistory.com/21<br/>

Setup might be really hard for first-timers.<br/>
But I'm pretty sure this would be good experience in future:)<br/>

## 4. Download caffe model and add to file
Download model file from [This link](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/blob/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel) and add to MultiPoseCV\opencvcameras\pose\coco

# If you completed the whole process above...
Run the program and Enjoy!!


감사합니다
