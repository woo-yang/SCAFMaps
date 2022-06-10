#pragma once

#include <Eigen/Dense>
#include "../common/mesh_type.h"

namespace alg {
	int mesh_optimization(
		const common::tri_mesh& mesh,
		common::ext_mesh& extend_mesh,
		int max_iter,
		std::function<void(const common::ext_mesh&)> call_back = nullptr);
}