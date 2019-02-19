#include "deal_2dpts.hpp"

void get_over01_idx(std::string path_over01_idx, Eigen::VectorXi &over01_idx) {
	puts("get idx z over -0.1");
	FILE *fp;
	fopen_s(&fp, path_over01_idx.c_str(), "r");
	int n;
	fscanf_s(fp, "%d", &n);
	printf("number of idx: %d\n", n);
	over01_idx.resize(n);
	for (int i = 0; i < n; i++)
		fscanf_s(fp, "%d", &over01_idx(i));
	fclose(fp);
}

void mesh_overlay_pic(std::string pic_path, Eigen::MatrixX2f &mesh2d_vtx) {
	puts("mesh covering...");
	cv::Mat img = cv::imread(pic_path.c_str());
	cv::Mat img_ini = img.clone();

	for (int i = 0; i < 10; i++)
		std::cout << mesh2d_vtx.row(i) << "\n";
	for (int i = 0; i < mesh2d_vtx.rows(); i++) {
		/*Point pos(
			width - 1 - (int)(round((y[i] - mi_y) / (ma_y - mi_y)*(width - 1))),
			height - 1 - (int)(round((x[i] - mi_x) / (ma_x - mi_x)*(height - 1))));
		img.at<Vec3f>(pos) = Vec3f(b[i], g[i], r[i]);*/
		cv::Point pos(mesh2d_vtx(i, 0), img.rows - mesh2d_vtx(i, 1));
		//printf("%d %.10f %.10f %d %d\n", i, mesh2d_vtx(i, 0), img.rows - mesh2d_vtx(i, 1), img.cols, img.rows);
		std::cout << pos << "\n";
		if (pos.inside(cv::Rect(0, 0, img.cols, img.rows)))
			img.at<cv::Vec3b>(pos) = cv::Vec3b(133, 180, 250);
		else
			printf("%d %.10f %.10f\n", i, mesh2d_vtx(i, 0), img.rows - mesh2d_vtx(i, 1));
	}
	//GaussianBlur(img, img, Size(3, 3), 0, 0);
	//blur(img, img, Size(10, 10));
	cv::imshow("test", img_ini);
	cv::imshow("result", img);
	cv::waitKey(0);
}


void get_face_hull_idx(std::vector<int> &face_hull_idx) {
	puts("getting face hull index");
	for (int i = 0; i < 15; i++) face_hull_idx.push_back(i);
	face_hull_idx.push_back(15); face_hull_idx.push_back(16); face_hull_idx.push_back(17);
	face_hull_idx.push_back(23); face_hull_idx.push_back(22); face_hull_idx.push_back(21);
}

bool bj_fld_fl[1000][1000];

void face_transfer(
	DataPoint &sc_data, DataPoint &tg_data,
	Eigen::MatrixX2f &sc_mesh2d_vtx, Eigen::MatrixX2f &tg_mesh2d_vtx,
	cv::Mat tg_face_hull_mask, cv::Mat ans) {

	puts("face transferring...");
	std::queue<std::pair<int, cv::Point> > que;
	while (!que.empty()) que.pop();

	memset(bj_fld_fl, 0, sizeof(bj_fld_fl));

	for (int i_v = 0, sz = tg_mesh2d_vtx.rows(); i_v < sz; i_v++) {
		printf("fc tf %d\n", i_v);
		cv::Point pos = cv::Point((int)tg_mesh2d_vtx(i_v, 0), (int)tg_mesh2d_vtx(i_v, 1));
		if (tg_face_hull_mask.at<cv::Vec3b>(pos) != cv::Vec3b(255, 255, 255)) continue;
		ans.at<cv::Vec3b>(pos) = sc_data.image.at<cv::Vec3b>(cv::Point((int)sc_mesh2d_vtx(i_v, 0), (int)sc_mesh2d_vtx(i_v, 1)));
		que.push(std::make_pair(i_v, cv::Point(0, 0)));
		bj_fld_fl[pos.x][pos.y] = 1;
	}
	puts("fc tf next");
	flood_fill(que, sc_data, tg_face_hull_mask, sc_mesh2d_vtx,tg_mesh2d_vtx, sc_data.eye_dis/tg_data.eye_dis, ans);

}
const int fdfl_dx[8] = { -1,-1,-1,0,0,1,1,1 };
const int fdfl_dy[8] = { -1,0,1,-1,1,-1,0,1 };
void flood_fill(
	std::queue<std::pair<int, cv::Point> > &que, DataPoint &sc_data, cv::Mat tg_face_hull_mask,
	Eigen::MatrixX2f &sc_mesh2d_vtx, Eigen::MatrixX2f &tg_mesh2d_vtx, float ratio, cv::Mat ans) {

	puts("flood filling...");
	while (!que.empty()) {
		int idx = que.front().first;
		int dx = que.front().second.x, dy = que.front().second.y;
		que.pop();
		printf("fdfl %d %d %d\n", idx, dx, dy);
		for (int i = 0; i < 8; i++) {
			int du = dx + fdfl_dx[i], dv = dy + fdfl_dy[i];
			cv::Point pos = cv::Point((int)tg_mesh2d_vtx(idx, 0) + du, (int)tg_mesh2d_vtx(idx, 1) + dv);
			if (bj_fld_fl[pos.x][pos.y]) continue;
			if (tg_face_hull_mask.at<cv::Vec3b>(pos) != cv::Vec3b(255, 255, 255)) continue;

			int sc_du = (int)(du* ratio + 0.5),sc_dv=(int)(dv*ratio+0.5);
			ans.at<cv::Vec3b>(pos) =
				sc_data.image.at<cv::Vec3b>(cv::Point((int)sc_mesh2d_vtx(idx, 0) + sc_du, (int)sc_mesh2d_vtx(idx, 1) + sc_dv));
			que.push(std::make_pair(idx,cv::Point(du, dv)));
			bj_fld_fl[pos.x][pos.y] = 1;
		}
	}
}
