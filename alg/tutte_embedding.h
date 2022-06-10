#include <Eigen/Dense>
#include "../common/mesh_type.h"

namespace alg {

	int tutte_embedding(
		const common::tri_mesh& mesh,
		const Eigen::VectorXi& bnd,
		common::uv_mesh& uv_mesh);

	int triangulate_scaffold(
		const common::uv_mesh& uv_mesh,
		const Eigen::VectorXi& bnd,
		common::ext_mesh& extend_mesh);
}