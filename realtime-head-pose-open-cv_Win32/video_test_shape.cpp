// https://github.com/lincolnhard/head-pose-estimation/blob/master/video_test_shape.cpp

#pragma once
#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include "pose_estimate.h"

#define FaceCENTER  0
#define FaceRIGHT   1
#define FaceLEFT    2
#define FaceUP      1
#define FaceDOWN    2
#define RollLEFT    1
#define RollRIGHT   2

//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

// ������ - ������
// ��ƼĿ ���̱⿡ ����� ������
// ��ƼĿ ����, ����ũ, ���ɿ���
cv::Mat img;
cv::Mat mask_img;
cv::Mat img_left;
cv::Mat mask_left;
cv::Mat img_right;
cv::Mat mask_right;
cv::Mat rect;

cv::Mat mosaic_temp;

// ��ƼĿ ���� ����
//int img_width;
//int img_height;

int sticker_start_X;
int sticker_start_Y;

// ����ũ resize ����
int resized_width;
int resized_height;

// �� ����
double posX;
double posY;
double posZ;

// ī�޶� â ����
int temp_width;
int temp_height;

// �� ���� ������ ��ǥ
cv::Point2d faceStart, faceEnd;

// ���� ��Ż�� Ȯ���ϱ� ���� ���� ����
int check_W;
int check_H;

// filter ������ ���� ON OFF ������
int sticker_bear = 0;
int mosaic_OnNOff = 0;

// mosaic_strength
int mosaic_strength = 1;

// filter ���� �Լ�
void trackbar_sticking_bear(int pos, void* userdata);
void trackbar_mosaicing(int pos, void* userdata);

// ��ƼĿ ���� ��Ż Ȯ�ο�
void check_point_out(cv::Point faceStart, cv::Point faceEnd); // ������ - ������

void modulate_ratio(cv::Point faceStart, cv::Point faceEnd);
void get_RoI(cv::Mat* src);
//void sticker_resizing(cv::Mat* src, cv::Mat* src_mask);
void sticking(cv::Mat* src, cv::Mat* src_mask);

void select_sticker_pose();

void get_bear();  // ������ - ������
void get_bear_Right();
void get_bear_Left();
//void modulate_roi(cv::Mat src, cv::Point faceStart, cv::Point faceEnd);   // ������ - ������
//void modulate_roi(cv::Mat src, int facePosX, int facePosY);   // ������ - ������

int faceSide = 0;
int faceUPDOWN = 0;
int faceRoll = 0;

int main() {
    //open cam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Unable to connect to camera" << std::endl;
        return EXIT_FAILURE;
    }

    // �̸� ���� ���� : �������� main loop �ȿ� �־���
    cv::Mat temp;

    // filter�� �����ų Ʈ���� ���� â ����
    cv::namedWindow("filters");
    cv::createTrackbar("bear", "filters", 0, 1, trackbar_sticking_bear, (void*)0);
    cv::createTrackbar("mosaic", "filters", 0, 10, trackbar_mosaicing, (void*)0);

    //Load face detection and pose estimation models (dlib).
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    //fill in cam intrinsics and distortion coefficients
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    std::vector<cv::Point3d> object_pts;
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

    //2D ref points(image coordinates), referenced from detected facial feature
    std::vector<cv::Point2d> image_pts;

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

    //reproject 3D points world coordinate axis to verify result pose
    std::vector<cv::Point3d> reprojectsrc;
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

    //reprojected 2D points
    std::vector<cv::Point2d> reprojectdst;
    reprojectdst.resize(8);

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

    //text on screen
    std::ostringstream outtext;

    // ���� ���� ����
    cap >> temp;
    temp_width = temp.cols;
    temp_height = temp.rows;

    //main loop
    while (1) {
        // Grab a frame
        cap >> temp;
        dlib::cv_image<dlib::bgr_pixel> cimg(temp);

        // Detect faces
        std::vector<dlib::rectangle> faces = detector(cimg);

        // Find the pose of each face
        if (faces.size() > 0) {
            //track features
            dlib::full_object_detection shape = predictor(cimg, faces[0]);

            /*
            * �� ���
            //draw features
            for (unsigned int i = 0; i < 68; ++i) {
                cv::circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
            }
            */

            //fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
            image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
            image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
            image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
            image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
            image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
            image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
            image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
            image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
            image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
            image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
            image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
            image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
            image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
            image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

            //calc pose
            cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

            //reproject
            cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

            //draw axis
            /*
            cv::line(temp, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[1], reprojectdst[2], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[2], reprojectdst[3], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[3], reprojectdst[0], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[4], reprojectdst[5], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[5], reprojectdst[6], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[6], reprojectdst[7], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[7], reprojectdst[4], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[0], reprojectdst[4], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[1], reprojectdst[5], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[2], reprojectdst[6], cv::Scalar(0, 0, 255));
            cv::line(temp, reprojectdst[3], reprojectdst[7], cv::Scalar(0, 0, 255));
            */

            /*
            *  �簢�� ��ǥ �˻�
            */
            std::string number;
            for (int i = 0; i < reprojectdst.size(); ++i) {
                number = (std::to_string(i));
                cv::putText(temp, number, reprojectdst[i], cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            }

            //calc euler angle
            cv::Rodrigues(rotation_vec, rotation_mat);
            cv::hconcat(rotation_mat, translation_vec, pose_mat);
            cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

            posX = euler_angle.at<double>(0); // ����, pitch
            posY = euler_angle.at<double>(1); // �¿�, yaw
            posZ = euler_angle.at<double>(2); // ����, roll

            /*
            cout << "reprojectdst[0] : " << reprojectdst[0] << endl;
            cout << "reprojectdst[0].x : " << reprojectdst[0].x << endl;
            cout << "reprojectdst[0].y : " << reprojectdst[0].y << endl << endl;;
            printf("f reprojectdst[0].x : %f\n", reprojectdst[0].x);
            printf("f reprojectdst[0].y : %f\n\n", reprojectdst[0].y);
            printf("casting(int) reprojectdst[0].x : %d\n", (int)reprojectdst[0].x);
            printf("casting(int) reprojectdst[0].y : %d\n\n", (int)reprojectdst[0].y);
            */

            // 0. ���ɿ����� �������� ������ ����
                // reprojectdst[0] �� reprojectdst[1]�� �߰����� ���
                // �Ҽ��� �Ʒ� ù��° �ڸ����� �ݿø��ߴٰ� �����Ͽ� +1
                // sticker�� �󱼿� �µ��� �»�� �𼭸��� �̵���Ŵ
            faceStart.x = ((int)reprojectdst[0].x + (int)reprojectdst[1].x) / 2 + 1 - sticker_start_X;
            faceStart.y = ((int)reprojectdst[0].y + (int)reprojectdst[0].y) / 2 + 1 - sticker_start_Y;
                // reprojectdst[6] �� reprojectdst[6]�� �߰����� ���
            faceEnd.x = ((int)reprojectdst[6].x + (int)reprojectdst[7].x) / 2;
            faceEnd.y = ((int)reprojectdst[6].y + (int)reprojectdst[7].y) / 2;

            /*
            cout << "faceStart : " << faceStart << endl;
            cout << "faceStart.x : " << faceStart.x << endl;
            cout << "faceStart.y : " << faceStart.y << endl << endl;;
            cout << "faceEnd : " << faceEnd << endl;
            cout << "faceEnd.x : " << faceEnd.x << endl;
            cout << "faceEnd.y : " << faceEnd.y << endl << endl;;
            */

            //cv::putText(temp, "S", cv::Point(facePosX, facePosY), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            cv::putText(temp, "S", faceStart, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
            cv::putText(temp, "E", faceEnd, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));

            // �� �¿� Ȯ��
            if (posY > 17) { // �¿� right
                faceSide = FaceRIGHT;
            }
            else if (posY < -17) { // �¿� left
                faceSide = FaceLEFT;
            }
            else { // �¿� center
                faceSide = FaceCENTER;
            }

            // �� ���� Ȯ��
            /*
            * ���� ķ�� �� ��
            if (posX < -5) { // up
                faceUPDOWN = FaceUP;
            }
            else if (posX > 7) { // down
                faceUPDOWN = FaceDOWN;
            }
            else
                faceUPDOWN = FaceCENTER;
            */

            // �� ��ǻ�� ����ķ �� ��; ���ΰ�
            if (posX < -15) { // �¿� center ���� up
                faceUPDOWN = FaceUP;
            }
            else if (posX > -10) { // �¿� center ���� down
                faceUPDOWN = FaceDOWN;
            }
            else {
                faceUPDOWN = FaceCENTER;
            }

            // �� �� Ȯ��
            if (posZ < -20) {
                faceRoll = RollLEFT;
            }
            else if (posZ > 20) {
                faceRoll = RollRIGHT;
            }
            else {
                faceRoll = FaceCENTER;
            }

            /* Ʈ���� ���� */
            if (sticker_bear) {
                //printf("sticker_bear\n");
                // 1.���ɿ��� ����� ��ü ����� ����� �ʰ� ����
                check_point_out(faceStart, faceEnd);
                // 2. ���ɿ����� ����� 1:1 ������ ����
                modulate_ratio(faceStart, faceEnd);
                // 3. ���ɿ��� ����
                get_RoI(&temp);

                // ȭ�� ������ ���� Ʈ���� �Լ� �ȿ� ���� �ʰ� ���� �б⸦ ��
                get_bear();
                get_bear_Left();
                get_bear_Right();

                // �� ���⿡ �°� ��ƼĿ�� ����
                select_sticker_pose();

                cv::imshow("img", img);
                //cv::imshow("img_left", img_left);
                //cv::imshow("img_right", img_right);
            }
            if (mosaic_OnNOff) {
                //printf("mosaic_OnNOff\n");
                    // 1.���ɿ��� ����� ��ü ����� ����� �ʰ� ����
                check_point_out(faceStart, faceEnd);
                    // 2. ���ɿ����� ����� 1:1 ������ ����
                modulate_ratio(faceStart, faceEnd);
                    // 3. ���ɿ��� ����
                get_RoI(&temp);

                // ������ũ ����
                cv::resize(rect, mosaic_temp, cv::Size(rect.rows / mosaic_strength, rect.cols / mosaic_strength));
                cv::resize(mosaic_temp, rect, cv::Size(rect.rows, rect.cols));
            }

            //show angle result
            outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);//���� pitch
            cv::putText(temp, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
            outtext.str("");
            outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);//�¿� yaw
            cv::putText(temp, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            outtext.str("");
            outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);//���� roll
            cv::putText(temp, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
            outtext.str("");

            image_pts.clear();
        }

        //press esc to end
        cv::imshow("demo", temp);
        unsigned char key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }

    return 0;
}

/* Ʈ���ٿ� ���� �Լ��� */
void trackbar_sticking_bear(int pos, void* userdata) {
    printf("trackbar_sticking_bear\n");

    sticker_bear = pos;
    if (pos != 0) {
        cv::setTrackbarPos("mosaic", "filters", 0);
    }
}

void trackbar_mosaicing(int pos, void* userdata) {
    printf("trackbar_mosaicing\n");

    mosaic_OnNOff = pos;
    mosaic_strength = pos;
    if (pos != 0) {
        cv::setTrackbarPos("bear", "filters", 0);
    }
}

/* ��ƼĿ�� �����ϴµ� �ʿ��� �Լ��� */
// roi�� ���� ��ǥ�� ���� ������ ����� ������ �̵���Ŵ
void check_point_out(cv::Point faceStart, cv::Point faceEnd) {
    printf("check_point_out\n");

    if (faceStart.x <= 0) {
        faceStart.x = 1;
    }
    if (faceStart.y <= 0) {
        faceStart.y = 1;
    }
    if (faceEnd.x >= temp_width){
        faceEnd.x = temp_width - 1;
    }
    if (faceEnd.y >= temp_height) {
        faceEnd.y = temp_height - 1;
    }
}

// roi�� width, height ������ 1:1�� ����
void modulate_ratio(cv::Point faceStart, cv::Point faceEnd) {
    printf("modulate_ratio\n");

    // �� ���� ���̸� roi ���̷� ����
    if (faceEnd.x < faceEnd.y) {
        resized_width = faceEnd.x - faceStart.x;
        resized_height = resized_width;
    }
    else {
        resized_height = faceEnd.y - faceStart.y;
        resized_width = resized_height;
    }

    // Ȯ�ο�
    printf("resized length : %d\n", resized_width);
    printf("resized length : %d\n\n", resized_height);
}

void get_RoI(cv::Mat* src) {
    printf("get_RoI\n");
    
    cv::Mat img = *src;

    // ���ɿ��� ����
    rect = img(cv::Rect(faceStart.x, faceStart.y, resized_width, resized_height));

    // Ȯ�ο�
    cv::rectangle(img, cv::Rect(faceStart.x, faceStart.y, resized_width, resized_height), cv::Scalar(0, 255, 0), 1);

    modulate_ratio(faceStart, faceEnd);
}
/*
void sticker_resizing(cv::Mat* img, cv::Mat* img_mask) {
    printf("sticker_resizing\n");
    cv::Mat src = *img;
    cv::Mat src_mask = *img_mask;

    cv::resize(src, src, cv::Size(resized_width, resized_height));
    cv::resize(src_mask, src_mask, cv::Size(resized_width, resized_height));
}
*/

void sticking(cv::Mat* img, cv::Mat* img_mask) {
    printf("sticking\n");
    cv::Mat src = *img;
    cv::Mat src_mask = *img_mask;
    // 4. ��ƼĿ�� ���ɿ��� ����� �°� ����
    //sticker_resizing(&src, &src_mask);
    cv::resize(src, src, cv::Size(resized_width, resized_height));
    cv::resize(src_mask, src_mask, cv::Size(resized_width, resized_height));

    //cv::imshow("src", src); // test
    // ���� ������ �̹��� �ٿ��ֱ�
    src.copyTo(rect, src_mask);
    //cv::imshow("test", rect); // test
}

void select_sticker_pose() {
    printf("select_sticker_pose\n");
    // ���Ʒ� Ȯ���� ������
    if (faceUPDOWN == FaceCENTER) { // ���Ʒ� center
        if (faceSide == FaceCENTER) { // �¿� center
            sticking(&img, &mask_img);
        }
        else if (faceSide == FaceLEFT) {    // �¿� LEFT
            sticking(&img_left, &mask_left);
        }
        else if (faceSide == FaceRIGHT) {   // �¿� RIGHT
            sticking(&img_right, &mask_right);
        }
    }
    else if (faceUPDOWN == FaceUP) { // ���Ʒ� UP
        if (faceSide == FaceCENTER) { // �¿� center
            sticking(&img, &mask_img);
        }
        else if (faceSide == FaceLEFT) {    // �¿� LEFT
            sticking(&img_left, &mask_left);
        }
        else if (faceSide == FaceRIGHT) {   // �¿� RIGHT
            sticking(&img_right, &mask_right);
        }
    }
    else if (faceUPDOWN == FaceDOWN) { // ���Ʒ� DOWN
        if (faceSide == FaceCENTER) { // �¿� center
            sticking(&img, &mask_img);
        }
        else if (faceSide == FaceLEFT) {    // �¿� LEFT
            sticking(&img_left, &mask_left);
        }
        else if (faceSide == FaceRIGHT) {   // �¿� RIGHT
            sticking(&img_right, &mask_right);
        }
    }
}

void get_bear() { // ������ - ������
    printf("get_bear\n");

    img = cv::imread("./sticker/bear/bear_400.jpg");
    mask_img = cv::imread("./sticker/bear/bear_400_mask.jpg", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        cerr << "bear load failed!" << endl;
        exit(0);
    }
    if (mask_img.empty()) {
        cerr << "bear_mask load failed!" << endl;
        exit(0);
    }

    sticker_start_X = 60;
    sticker_start_Y = 60;

    //img_width = img.cols;
    //img_height = img.rows;
}


void get_bear_Left() {
    printf("get_bear_Left\n");

    img_left = cv::imread("./sticker/bear/bear_Left.jpg");
    mask_left = cv::imread("./sticker/bear/bear_Left_mask.jpg", cv::IMREAD_GRAYSCALE);

    if (img_left.empty()) {
        cerr << "bear load failed!" << endl;
        exit(0);
    }
    if (mask_left.empty()) {
        cerr << "bear_mask load failed!" << endl;
        exit(0);
    }

    sticker_start_X = 60;
    sticker_start_Y = 60;

    //img_width = img_left.cols;
    //img_height = img_left.rows;
}

void get_bear_Right() {
    printf("get_bear_Right\n");

    img_right = cv::imread("./sticker/bear/bear_Right.jpg");
    mask_right = cv::imread("./sticker/bear/bear_Right_mask.jpg", cv::IMREAD_GRAYSCALE);

    if (img_right.empty()) {
        cerr << "bear load failed!" << endl;
        exit(0);
    }
    if (mask_right.empty()) {
        cerr << "bear_mask load failed!" << endl;
        exit(0);
    }

    sticker_start_X = 60;
    sticker_start_Y = 60;

    //img_width = img_right.cols;
    //img_height = img_right.rows;
}

/*
void modulate_roi(cv::Mat src, int facePosX, int facePosY) {   // ������ - ������
    check_W = facePosX + resized_height;
    check_H = facePosY + resized_width;

    if (check_W >= temp_width || check_H >= temp_height) { // width or height�� ������ ��� ��
        // �� ���� ���̷� ������ ����
        if (check_W < check_H) {
            // width�� �����ϰ� 1:1 ������ ����
            resized_width = temp_width - facePosX - 1;
            resized_height = resized_width;
        }
        else {
            // height�� �����ϰ� 1:1 ������ ����
            resized_height = temp_height - facePosY - 1;
            resized_width = resized_height;
        }
    }
    else { // ����� ���� ��
        // ���� ���̷� �ٽ� ��
        check_W = facePosX + img_width;
        check_H = facePosY + img_height;

        if (check_W >= temp_width || check_H >= temp_height) {
            if (check_W < check_H) {
                resized_width = temp_width - facePosX - 1;
                resized_height = resized_width;
            }
            else {
                resized_height = temp_height - facePosY - 1;
                resized_width = resized_height;
            }
        }       // ���� ���̷ε� ������ ����� ���� ��
        else if (img_width != resized_width) { // resize�� ������
            // �̹��� �ٽ� ���� : ȭ���� �����ϱ� ����
            if (sticker_bear) {
                get_bear();
            }
        }
    }

    // 1. ���ɿ��� ����
    rect = src(cv::Rect(facePosX, facePosY, resized_width, resized_height));

    cv::rectangle(src, cv::Rect(facePosX, facePosY, resized_width, resized_height), cv::Scalar(0, 255, 0), 1);

    // 2. ���� ������ �´� ������� ����ũ ������ ����
    cv::resize(img, img, cv::Size(resized_width, resized_height));
    cv::resize(mask, mask, cv::Size(resized_width, resized_height));

    // 3. ���� ������ �̹��� ����
    img.copyTo(rect, mask);

    // Ȯ�ο�
    cv::imshow("rect", rect);
    cv::imshow("img", img);
}
*/