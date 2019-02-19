#include "deal_2dpts.hpp"


const std::string path_over01_idx = "over01_idx.txt";

#ifdef win64
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
const std::string sc_path = "D:\\sydney\\first\\data_me\\test_lv\\fw\\Tester_28\\TrainingPose/pose_1.";
const std::string tg_path = "D:\\sydney\\first\\data_me\\test_lv\\fw\\Tester_48\\TrainingPose/pose_5.";
#endif // win64
#ifdef linux
std::string bldshps_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_77.lv";
const std::string sc_path = "/home/weiliu/fitting_dde/fitting_coef_l1_l2/fw/Tester_28/TrainingPose/pose_1.";
const std::string tg_path = "/home/weiliu/fitting_dde/fitting_coef_l1_l2/fw/Tester_28/TrainingPose/pose_5.";
#endif // linux



void transfer_pic2pic(Eigen::MatrixXf &bldshps, Eigen::VectorXi &over01_idx,std::vector<int> &face_hull_idx) {

	DataPoint sc_data, tg_data;

	load_lv(sc_path + "lv", sc_data); load_lv(tg_path + "lv", tg_data);
	sc_data.image = cv::imread(sc_path + "jpg"); tg_data.image = cv::imread(tg_path + "jpg");
	//puts("calculating landmarks...");
	//sc_data.landmarks.resize(G_land_num);
	//for (int i_v = 0; i_v < G_land_num; i_v++) {
	//	sc_data.landmarks[i_v].x = sc_data.land_2d(i_v, 0) + sc_data.center(0);
	//	sc_data.landmarks[i_v].y = sc_data.image.rows - sc_data.land_2d(i_v, 1) - sc_data.center(1);
	//}
	//puts("calculating tg landmarks...");
	//tg_data.landmarks.resize(G_land_num);
	//for (int i_v = 0; i_v < G_land_num; i_v++) {

	//	tg_data.landmarks[i_v].x = tg_data.land_2d(i_v, 0) + tg_data.center(0);
	//	tg_data.landmarks[i_v].y = tg_data.image.rows - tg_data.land_2d(i_v, 1) - tg_data.center(1);
	//}
	cal_data_landmarks(sc_data, bldshps); cal_data_landmarks(tg_data, bldshps);



	std::vector<cv::Point> sc_face_hull_pts, tg_face_hull_pts;
	for (int i = 0, sz = face_hull_idx.size(); i < sz; i++) sc_face_hull_pts.push_back(sc_data.landmarks[face_hull_idx[i]]);
	for (int i = 0, sz = face_hull_idx.size(); i < sz; i++) tg_face_hull_pts.push_back(tg_data.landmarks[face_hull_idx[i]]);

	puts("calculating mask...");
	cv::Mat sc_face_hull_mask = sc_data.image.clone();
	cv::Mat tg_face_hull_mask = tg_data.image.clone();

	//std::cout << face_hull_pts << "\n";

	cv::fillConvexPoly(sc_face_hull_mask, &sc_face_hull_pts[0], face_hull_idx.size(), cv::Scalar(255, 255, 255));
	cv::fillConvexPoly(tg_face_hull_mask, &tg_face_hull_pts[0], face_hull_idx.size(), cv::Scalar(255, 255, 255));
	cv::imshow("mask_sc", sc_face_hull_mask);
	cv::imshow("mask_tg", tg_face_hull_mask);
	//cv::waitKey(0);

	

	Eigen::MatrixX2f sc_mesh2d_vtx_i, tg_mesh2d_vtx_i;

	get_mesh2d_i(sc_data, sc_mesh2d_vtx_i, path_over01_idx, bldshps, over01_idx);
	tg_data.user = sc_data.user;
	get_mesh2d_i(tg_data, tg_mesh2d_vtx_i, path_over01_idx, bldshps, over01_idx);

	cv::Mat result_test = tg_face_hull_mask.clone();
	for (int i = 0; i < tg_mesh2d_vtx_i.rows(); i++) {
		cv::Point pos = cv::Point(tg_mesh2d_vtx_i(i, 0), tg_mesh2d_vtx_i(i, 1));
		result_test.at<cv::Vec3b>(pos) = cv::Vec3b(0, 0, 255);
	}
	cv::imshow("result_test", result_test);
	cv::waitKey();


	cv::Mat result = tg_face_hull_mask.clone();
	face_transfer(sc_data, tg_data, sc_mesh2d_vtx_i, tg_mesh2d_vtx_i, tg_face_hull_mask, result);
	//void face_transfer(
	//	DataPoint &sc_data, DataPoint &tg_data,
	//	Eigen::MatrixX2f &sc_mesh2d_vtx_i, Eigen::MatrixX2f &tg_mesh2d_vtx_i,
	//	cv::Mat tg_face_hull_mask, cv::Mat ans)
	cv::imshow("result", result);
	cv::waitKey(0);
}

int main()
{
	Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
	load_bldshps(bldshps, bldshps_path);

	Eigen::VectorXi over01_idx;
	get_over01_idx(path_over01_idx, over01_idx);

	std::vector<int> face_hull_idx;
	get_face_hull_idx(face_hull_idx);

	transfer_pic2pic(bldshps, over01_idx, face_hull_idx);
	
	return 0;
}
//cv::Mat img = cv::imread((path + "jpg").c_str());
	//std::cout << img.type() << "\n";
	//std::cout << img.at<cv::Vec3b>(1, 2) << "\n";


	//std::cout << over01_idx.transpose() << "\n";




//	mesh_overlay_pic(path + "jpg",mesh2d_vtx_i);

/*
grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'
*/