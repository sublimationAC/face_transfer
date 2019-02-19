#pragma once
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>
#include <ctype.h>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <queue>

#define normalization
#define win64
//#define linux

#define same_id

const int G_land_num = 74;
const int G_train_pic_id_num = 3300;
const int G_nShape = 47;
const int G_nVerts = 11510;
const int G_nFaces = 11540;
const int G_test_num = 77;
const int G_iden_num = 77;
const int G_inner_land_num = 59;
const int G_line_num = 50;
const int G_jaw_land_num = 20;

const int G_left_eye_idx = 29;
const int G_right_eye_idx = 33;


struct Target_type {
	Eigen::VectorXf exp;
	Eigen::RowVector3f tslt;
	Eigen::Matrix3f rot;
	Eigen::MatrixX2f dis;

};

struct DataPoint
{
	cv::Mat image;
	std::vector<cv::Point2d> landmarks;
	//std::vector<cv::Point2d> init_shape;
	Target_type shape;
	Eigen::VectorXf user;
	Eigen::RowVector2f center;
	Eigen::MatrixX2f land_2d;
#ifdef posit
	float f;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif

	Eigen::VectorXi land_cor;
	float eye_dis;
};

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name);

void load_lv(std::string name, DataPoint &temp);

void get_mesh2d_i(
	DataPoint &data, Eigen::MatrixX2f &mesh_2d_i, std::string path_over01_idx,
	Eigen::MatrixXf &bldshps, Eigen::VectorXi &over01_idx);

double dis_cv_pt(cv::Point2d pointO, cv::Point2d pointA);

void cal_data_landmarks(DataPoint &data, Eigen::MatrixXf &bldshps);