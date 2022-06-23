#pragma once
#include "../common/mesh_type.h"

#include <set>

namespace alg {

	int generate_seam(
		const common::tri_mesh& mesh,
		std::set<std::pair<int, int>>& edge_set,
		const int r = 3, const double percentage = 0.98);

}