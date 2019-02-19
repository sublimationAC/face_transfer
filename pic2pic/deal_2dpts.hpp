#pragma once
#include "get_mesh.hpp"

void get_over01_idx(std::string path_over01_idx, Eigen::VectorXi &over01_idx);
void mesh_overlay_pic(std::string pic_path, Eigen::MatrixX2f &mesh2d_vtx);

void get_face_hull_idx(std::vector<int> &face_hull_idx);

void face_transfer(
	DataPoint &sc_data, DataPoint &tg_data,
	Eigen::MatrixX2f &sc_mesh2d_vtx, Eigen::MatrixX2f &tg_mesh2d_vtx,
	cv::Mat tg_face_hull_mask, cv::Mat ans);

void flood_fill(
	std::queue<std::pair<int, cv::Point> > &que, DataPoint &sc_data, cv::Mat tg_face_hull_mask,
	Eigen::MatrixX2f &sc_mesh2d_vtx, Eigen::MatrixX2f &tg_mesh2d_vtx, float ratio, cv::Mat ans);