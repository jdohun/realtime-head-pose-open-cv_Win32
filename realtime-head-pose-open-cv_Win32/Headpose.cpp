// https://github.com/lincolnhard/head-pose-estimation/blob/master/video_test_shape.cpp

#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include "pose_estimate.h"

// #include <string> // 사각형 좌표 찾을 때 사용 했음

//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

// 스티커 붙이기에 사용할 변수들
// 스티커 원본, 마스크, 관심영역
cv::Mat img;
cv::Mat mask;
cv::Mat rect;

// 스티커 원본 길이
int img_width;
int img_height;

// 마스크 resize 길이
int resized_width;
int resized_height;

// 얼굴 각도
double posX;
double posY;
double posZ;

// 카메라 창 길이
int temp_width;
int temp_height;

// 얼굴 시작 지점의 좌표
// reprojectdst[0],[1]의 x,y 값을 (int)로 캐스팅하여 담아놓기 위한 변수
// : 캐스팅하지 않으면 엉뚱한 값이 나옴
int facePosX;
int facePosY;

// 영역 이탈을 확인하기 위한 길이
int check_W;
int check_H;

void get_rabbit() {
    img = cv::imread("rabbit.png");
    mask = cv::imread("rabbit_mask.jpg", cv::IMREAD_GRAYSCALE);

    img_width = img.cols;
    img_height = img.rows;

    resized_width = img_width;
    resized_height = img_height;
}

void sticker_rabbit_ON(cv::Mat src, int facePosX, int facePosY) {
    check_W = facePosX + resized_height;
    check_H = facePosY + resized_width;

    if (check_W >= temp_width || check_H >= temp_height) { // width or height가 범위를 벗어날 때
        // 더 작은 길이로 사이즈 조절
        if (check_W < check_H) {
            // width를 조절하고 1:1 비율을 유지
            resized_width = temp_width - facePosX - 1;
            resized_height = resized_width;
        }
        else {
            // height를 조절하고 1:1 비율을 유지
            resized_height = temp_height - facePosY - 1;
            resized_width = resized_height;
        }
    }
    else { // 벗어나지 않을 때
        // 원본 길이로 다시 비교
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
        }       // 원본 길이로도 범위를 벗어나지 않을 때
        else if (img_width != resized_width) { // resize를 했으면
            // 이미지 다시 읽음 : 화질을 유지하기 위해
            img = cv::imread("rabbit.png");
            mask = cv::imread("rabbit_mask.jpg", cv::IMREAD_GRAYSCALE);

            // 원본 길이로 설정
            resized_width = img_width;
            resized_height = img_height;
        }
    }

    // 1. 관심영역 추출
    rect = src(cv::Rect(facePosX, facePosY, resized_width, resized_height));

    // 2. 관심 영역에 맞는 사이즈로 마스크 사이즈 변경
    cv::resize(img, img, cv::Size(resized_width, resized_height));
    cv::resize(mask, mask, cv::Size(resized_width, resized_height));

    cv::imshow("rect", rect);
    // 3. 관심 영역에 이미지 복사
    img.copyTo(rect, mask);
    cv::imshow("img", img);
}

int main() {
    //open cam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Unable to connect to camera" << std::endl;
        return EXIT_FAILURE;
    }
   
    get_rabbit();
    
    // 미리 변수 선언 : 기존에는 main loop 안에 있었음
    cv::Mat temp;

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
    
    // 원본 길이 저장
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
            * 점 찍기
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
            *  사각형 좌표 검색
            std::string number;
            for (int i = 0; i < reprojectdst.size(); ++i) {
                number = (std::to_string(i));
                cv::putText(temp, number, reprojectdst[i], cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            }
            */

            //calc euler angle
            cv::Rodrigues(rotation_vec, rotation_mat);
            cv::hconcat(rotation_mat, translation_vec, pose_mat);
            cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

            posX = euler_angle.at<double>(0); // 상하, pitch
            posY = euler_angle.at<double>(1); // 좌우, yaw
            posZ = euler_angle.at<double>(2); // 기울기, roll

            cout << "reprojectdst[0] : " << reprojectdst[0] << endl;
            cout << "reprojectdst[0].x : " << reprojectdst[0].x << endl;
            cout << "reprojectdst[0].y : " << reprojectdst[0].y << endl << endl;;
            printf("f reprojectdst[0].x : %f\n", reprojectdst[0].x);
            printf("f reprojectdst[0].y : %f\n\n", reprojectdst[0].y);
            printf("casting(int) reprojectdst[0].x : %d\n", (int)reprojectdst[0].x);
            printf("casting(int) reprojectdst[0].y : %d\n\n", (int)reprojectdst[0].y);

            // 0. 관심영역 사이즈가 전체 사이즈를 벗어나지 않게 조정
            // reprojectdst[0] 과 reprojectdst[1]의 중간지점 사용
            // 소수점 아래 첫번째 자리에서 반올림했다고 가정하여 +1
            facePosX = ((int)reprojectdst[0].x + (int)reprojectdst[1].x) / 2 + 1 - 70;
            facePosY = ((int)reprojectdst[0].y + (int)reprojectdst[0].y) / 2 + 1 - 60;
            
            cv::putText(temp, "O", cv::Point(facePosX, facePosY), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));

            // 스티커 붙이기 
            sticker_rabbit_ON(temp, facePosX, facePosY);

            if (posY > 10) { // 좌우 right
                cv::putText(temp, "right", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            }
            else if (posY < -10) { // 좌우 left
                cv::putText(temp, "left", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            }
            else { // 좌우 center
                /*
                * 정면 캠이 될 때
                if (posX < -5) { // up
                    cv::putText(temp, "center up", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                }
                else if (posX > 7) { // down
                    cv::putText(temp, "center down", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                }
                else
                    cv::putText(temp, "center center", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                */

                // 내 컴퓨터 정면캠 안 됨; ㅄ인가
                if (posX < -15) { // up
                    cv::putText(temp, "center up", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                }
                else if (posX > -10) { // down
                    cv::putText(temp, "center down", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                }
                else {
                    cv::putText(temp, "center center", cv::Point(300, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
                }
            }

            //show angle result
            outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);//상하 pitch
            cv::putText(temp, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
            outtext.str("");
            outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);//좌우 yaw
            cv::putText(temp, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
            outtext.str("");
            outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);//기울기 roll
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