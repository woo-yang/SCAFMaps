#include <Eigen/Dense>

#include <igl/boundary_loop.h>
#include <igl/cut_mesh.h>

#include <GWindow.h>
#include "../common/mesh_type.h"
#include "../common/mesh_util.h"
#include "../alg/mesh_optimization.h"
#include "../alg/seam_generation.h"
#include "../io/obj_io.h"

int scaf_maps(int argc, char* argv[]) {

	common::tri_mesh mesh;
	io::read_obj(argv[1], mesh.V_, mesh.F_);
	std::string path = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of('.'));

	std::set<std::pair<int, int>> seam;
	alg::generate_seam(mesh, seam);
	Eigen::MatrixXi seam_flag = Eigen::MatrixXi::Constant(mesh.V_.cols(), mesh.V_.cols(), 0);
	int i = 0;
	for (const auto& edge:seam) {
		seam_flag(edge.first, edge.second) =
		seam_flag(edge.second, edge.first) = 1;
		i++;
	}
	Eigen::MatrixXi C = Eigen::MatrixXi::Constant(mesh.F_.cols(), 3, 0);
	for (int i = 0; i < mesh.F_.cols(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (seam_flag(mesh.F_(j, i), mesh.F_((j + 1) % 3, i)))
				C(i, j) = 1;
		}
	}
	Eigen::MatrixX3d mesh_V = mesh.V_.transpose(), V_cut;
	Eigen::MatrixX3i mesh_F = mesh.F_.transpose(), F_cut;
	igl::cut_mesh(mesh_V, mesh_F, C, V_cut, F_cut);
	io::save_obj((path + "_with_seam.obj").c_str(), V_cut.transpose(), F_cut.transpose());

	common::tri_mesh mesh_seam;
	mesh_seam.V_ = V_cut.transpose();
	mesh_seam.F_ = F_cut.transpose();

	common::uv_mesh uv_mesh;
	common::ext_mesh extend_mesh;
	Eigen::VectorXi bound_id;
	igl::boundary_loop(mesh_seam.F_.transpose(), bound_id);

	alg::tutte_embedding(mesh_seam, bound_id, uv_mesh);
	io::save_obj((path + "_tutte_uv.obj").c_str(), mesh_seam.V_, mesh_seam.F_, uv_mesh.V_);
	alg::triangulate_scaffold(uv_mesh, bound_id, extend_mesh);

	YRender::GWindow window("scaf_maps");

	std::function<void(const common::ext_mesh&)> update
		= [&](const common::ext_mesh& m) {
		window.clear_object();
		Eigen::Matrix3Xd V;
		common::convert_uv_to_3d(m.V_, V);
		int id1 = window.push_object(V, m.F_.leftCols(m.mf_), YRender::Y_TRIANGLES, true);
		window.set_object_color(id1, 0, 1, 0.5);
		window.set_object_polygon_mode(id1, YRender::Y_LINE_AND_FILL);

		int id2 = window.push_object(V, m.F_.rightCols(m.sf_), YRender::Y_TRIANGLES, true);
		window.set_object_polygon_mode(id2, YRender::Y_LINE_AND_FILL);
		window.refresh();
	};
	
	update(extend_mesh);
	alg::mesh_optimization(mesh_seam, extend_mesh, 500 ,update);

	//write obj uv
	Eigen::Matrix3Xd V;
	common::convert_uv_to_3d(extend_mesh.V_.leftCols(mesh_seam.V_.cols()), V);

	io::save_obj((path + "_uv.obj").c_str(), V, extend_mesh.F_.leftCols(extend_mesh.mf_));
	io::save_obj((path + "_scaf_uv.obj").c_str(), mesh_seam.V_, mesh_seam.F_, extend_mesh.V_.leftCols(mesh_seam.V_.cols()));
	window.show();

	return 1;
}
