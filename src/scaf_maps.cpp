#include <Eigen/Dense>
#include <igl/boundary_loop.h>
#include <GWindow.h>

#include "../common/mesh_type.h"
#include "../common/mesh_util.h"
#include "../alg/tutte_embedding.h"
#include "../alg/mesh_optimization.h"
#include "../io/obj_io.h"

int scaf_maps(int argc, char* argv[]) {

	common::tri_mesh mesh;
	io::read_obj(argv[1], mesh.V_, mesh.F_);

	common::uv_mesh uv_mesh;
	common::ext_mesh extend_mesh;
	Eigen::VectorXi bound_id;
	igl::boundary_loop(mesh.F_.transpose(), bound_id);

	alg::tutte_embedding(mesh, bound_id, uv_mesh);
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
	alg::mesh_optimization(mesh, extend_mesh, 500 ,update);

	//write obj uv
	Eigen::Matrix3Xd V;
	common::convert_uv_to_3d(extend_mesh.V_.leftCols(mesh.V_.cols()), V);
	std::string path = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of('.'));

	io::save_obj((path + "_uv.obj").c_str(), V, extend_mesh.F_.leftCols(extend_mesh.mf_));
	io::save_obj((path + "_tutte_uv.obj").c_str(), mesh.V_, mesh.F_, uv_mesh.V_);
	io::save_obj((path + "_scaf_uv.obj").c_str(), mesh.V_, mesh.F_, extend_mesh.V_.leftCols(mesh.V_.cols()));
	window.show();

	return 1;
}
