#include "get_mesh.hpp"

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name) {

	puts("loading blendshapes...");
	std::cout << name << std::endl;
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	for (int i = 0; i < G_iden_num; i++) {
		for (int j = 0; j < G_nShape*G_nVerts * 3; j++)
			fread(&bldshps(i, j), sizeof(float), 1, fp);
	}
	fclose(fp);
}


void load_lv(std::string name, DataPoint &temp) {
	std::cout << "load coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	
	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);
	std::cout << temp.user << "\n";
	//system("pause");
	temp.land_2d.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
	}


	fread(&temp.center(0), sizeof(float), 1, fp);
	fread(&temp.center(1), sizeof(float), 1, fp);

	temp.shape.exp.resize(G_nShape);
	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fread(&temp.shape.exp(i_shape), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);

	temp.land_cor.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&temp.land_cor(i_v), sizeof(int), 1, fp);

	temp.s.resize(2, 3);
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fread(&temp.s(i, j), sizeof(float), 1, fp);

	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}
	std::cout << temp.shape.dis << "\n";
	//system("pause");
	fclose(fp);
	puts("load successful!");
}

float cal_3d_vtx(
	Eigen::MatrixXf &bldshps,
	Eigen::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;

	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			if (i_shape == 0)
				ans += exp(i_shape)*user(i_id)
				*bldshps(i_id, vtx_idx * 3 + axis);
			else
				ans += exp(i_shape)*user(i_id)
				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
	return ans;
}

void get_mesh2d_i(
	DataPoint &data, Eigen::MatrixX2f &mesh_2d_i, std::string path_over01_idx,
	Eigen::MatrixXf &bldshps,Eigen::VectorXi &over01_idx) {

	puts("calculating 2d mesh...");

	//printf("over01_idx.size() %d %d\n", over01_idx.size(), over01_idx.rows());
	mesh_2d_i.resize(over01_idx.size(), 2);

	Eigen::VectorXf user = data.user;
	Eigen::VectorXf exp = data.shape.exp;
	for (int i_idx = 0; i_idx < over01_idx.size(); i_idx++) {
		int i_v = over01_idx(i_idx);
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, user, exp, i_v, axis);
		mesh_2d_i.row(i_idx) = ((data.s*((data.shape.rot) * v)).transpose() +data.shape.tslt.block(0,0,1,2));
		mesh_2d_i(i_idx, 1) = data.image.rows - mesh_2d_i(i_idx, 1);
	}
	puts("calculating 2d mesh successful...");
}

double dis_cv_pt(cv::Point2d pointO, cv::Point2d pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);

	return distance;
}

void cal_data_landmarks(DataPoint &data, Eigen::MatrixXf &bldshps) {
	puts("calculating landmarks...");
	Eigen::VectorXf user = data.user;
	Eigen::VectorXf exp = data.shape.exp;
	data.landmarks.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, user, exp, data.land_cor(i_v), axis);
		v.transpose().block(0,0,1,2) = (data.s*((data.shape.rot) * v)).transpose() + data.shape.tslt.block(0, 0, 1, 2);
		data.landmarks[i_v].x = v(0);
		data.landmarks[i_v].y = data.image.rows - v(1);
	}
	data.eye_dis = dis_cv_pt(data.landmarks[G_left_eye_idx], data.landmarks[G_right_eye_idx]);
}


//g++ -Wall -std=c++11 `pkg-config --cflags opencv` -o deal main.cpp `pkg-config --libs opencv`

//void lv2mesh(std::string path, Eigen::MatrixXf &bldshps, std::string suffix[34469]) {
//
//
//	struct dirent **namelist;
//	int n;
//	n = scandir(path.c_str(), &namelist, 0, alphasort);
//	if (n < 0)
//	{
//		std::cout << "scandir return " << n << "\n";
//		perror("Cannot open .");
//		exit(1);
//	}
//	else
//	{
//		int index = 0;
//		struct dirent *dp;
//		while (index < n)
//		{
//			dp = namelist[index];
//			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
//			if (dp->d_name[0] == '.') {
//				free(namelist[index]);
//				index++;
//				continue;
//			}
//			if (dp->d_type == DT_DIR) {
//				lv2mesh(path + "/" + dp->d_name,bldshps,suffix);
//			}
//			else {
//				int len = strlen(dp->d_name);
//				if (dp->d_name[len - 1] == 'v' && dp->d_name[len - 2] == 'l') {
//					////	
//					std::string p = path + "/" + dp->d_name;
//					DataPoint data;
//					load_lv(p,data);
//					solve(data,bldshps,suffix, p.substr(0, p.find(".lv")) + "_0norm.obj");
//					std::cout << "cnt" << cnt << "\n";
//					cnt++;
//				}
//			}
//			free(namelist[index]);
//			index++;
//		}
//		free(namelist);
//	}
//
//}

//void lv2mesh_same_id(std::string path, Eigen::MatrixXf &exp_r_t_all_matrix, std::string suffix[34469]) {
//	struct dirent **namelist;
//	int n;
//	n = scandir(path.c_str(), &namelist, 0, alphasort);
//	if (n < 0)
//	{
//		std::cout << "scandir return " << n << "\n";
//		perror("Cannot open .");
//		exit(1);
//	}
//	else
//	{
//		int index = 0;
//		struct dirent *dp;
//		while (index < n)
//		{
//			dp = namelist[index];
//			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
//			if (dp->d_name[0] == '.') {
//				free(namelist[index]);
//				index++;
//				continue;
//			}
//			if (dp->d_type == DT_DIR) {
//				lv2mesh(path + "/" + dp->d_name, bldshps, suffix);
//			}
//			else {
//				int len = strlen(dp->d_name);
//				if (dp->d_name[len - 1] == 'v' && dp->d_name[len - 2] == 'l') {
//					////	
//					std::string p = path + "/" + dp->d_name;
//					DataPoint data;
//					load_lv(p, data);
//					solve_same_id(data, exp_r_t_all_matrix, suffix, p.substr(0, p.find(".lv")) + "_0norm.obj");
//					std::cout << "cnt" << cnt << "\n";
//					cnt++;
//				}
//			}
//			free(namelist[index]);
//			index++;
//		}
//		free(namelist);
//	}
//
//}

//void solve(DataPoint &data, Eigen::MatrixXf &bldshps, std::string suffix[34469], std::string name) {
//
//
//	puts("calculating and saving mesh...");
//	std::cout << "save obj name:" << name << "\n";
//	Eigen::MatrixX3f mesh(G_nVerts, 3);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf exp = data.shape.exp;
//	for (int i_v = 0; i_v < G_nVerts; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, exp, i_v, axis);
//		mesh.row(i_v) = ((data.shape.rot) * v).transpose();
//	}
//
//}
//int cnt = 0;
//void cal_exp_r_t_all_matrix(
//	Eigen::MatrixXf &bldshps, DataPoint &data, Eigen::MatrixXf &result) {
//
//	puts("prepare exp_point matrix for bfgs/ceres...");
//	result.resize(G_nShape, 3 * G_nVerts);
//
//	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//		for (int i_v = 0; i_v < G_nVerts; i_v++) {
//			Eigen::Vector3f V;
//			V.setZero();
//			for (int j = 0; j < 3; j++)
//				for (int i_id = 0; i_id < G_iden_num; i_id++)
//					if (i_shape == 0)
//						V(j) += data.user(i_id)*bldshps(i_id, i_v * 3 + j);
//					else
//						V(j) += data.user(i_id)*
//						(bldshps(i_id, i_shape*G_nVerts * 3 + i_v * 3 + j) - bldshps(i_id, i_v * 3 + j));
//
//			for (int j = 0; j < 3; j++)
//				result(i_shape, i_v * 3 + j) = V(j);
//		}
//#ifdef deal_64
//	result.block(0, 64 * 3, G_nShape, 3).array() = (result.block(0, 59 * 3, G_nShape, 3).array() + result.block(0, 62 * 3, G_nShape, 3).array()) / 2;
//#endif // deal_64
//}
//
//void solve_same_id(DataPoint &data, Eigen::MatrixXf &exp_r_t_all_matrix, std::string suffix[34469], std::string name) {
//
//
//	puts("calculating and saving mesh...");
//	std::cout << "save obj name:" << name << "\n";
//	Eigen::MatrixX3f mesh(G_nVerts, 3);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf exp = data.shape.exp;
//	for (int i_v = 0; i_v < G_nVerts; i_v++) {
//		Eigen::Vector3f v;
//		v.setZero();
//		for (int axis = 0; axis < 3; axis++)
//			for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//				v(axis) += data.shape.exp(i_shape)*exp_r_t_all_matrix(i_shape, i_v * 3 + axis);
//		mesh.row(i_v) = ((data.shape.rot) * v).transpose();
//	}
//}