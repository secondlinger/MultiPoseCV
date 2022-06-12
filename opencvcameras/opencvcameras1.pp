#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define COCO

#ifdef COCO
const int POSE_PAIRS[17][2] =
{
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
};

string protoFile = "pose/coco/pose_deploy_linevec.prototxt";
string weightsFile = "pose/coco/pose_iter_440000.caffemodel";


int nPoints = 18;
#endif


string device = "gpu";
string videoFile = "sample_video.mp4";



int inWidth = 368;
int inHeight = 368;
float thresh = 0.01;
double t = 0;
Mat inpBlob, output;
int H, W;
int frameWidth, frameHeight;
vector<Point> points(nPoints);
Net net;

Mat makebone(Mat frame) {
    inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

    net.setInput(inpBlob);

    output = net.forward();

    H = output.size[2];
    W = output.size[3];

    // find the position of the body parts

    for (int n = 0; n < nPoints; n++)
    {
        // Probability map of corresponding body's part.
        Mat probMap(H, W, CV_32F, output.ptr(0, n));

        Point2f p(-1, -1);
        Point maxLoc;
        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        if (prob > thresh)
        {
            p = maxLoc;
            p.x *= (float)frameWidth / W;
            p.y *= (float)frameHeight / H;

            circle(frame, cv::Point((int)p.x, (int)p.y), 8, Scalar(0, 255, 255), -1);
        }
        points[n] = p;
    }

    int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;

        line(frame, partA, partB, Scalar(0, 255, 255), 3);
        circle(frame, partA, 8, Scalar(0, 0, 255), -1);
        circle(frame, partB, 8, Scalar(0, 0, 255), -1);
    }
    return frame;
}


int main(void)
{


    cv::VideoCapture cap(videoFile);

    Mat frame;
    frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

    net = readNetFromCaffe(protoFile, weightsFile);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);


    while (1)
    {
        cap >> frame;
        cap >> frame;
        cap >> frame;
        cap >> frame;
        frame = makebone(frame);

        imshow("Output-Skeleton", frame);
        waitKey(1);
    }
    // When everything done, release the video capture and write object
    cap.release();

    return 0;
}