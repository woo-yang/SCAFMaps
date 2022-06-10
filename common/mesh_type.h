#pragma once

#include <Eigen/Dense>

namespace common {
	struct tri_mesh
	{
		Eigen::Matrix3Xd V_;
		Eigen::Matrix3Xi F_;
	};

	struct uv_mesh
	{
		Eigen::Matrix2Xd V_;
		Eigen::Matrix3Xi F_;
	};

	struct ext_mesh
	{
		Eigen::Matrix2Xd V_;
		Eigen::Matrix3Xi F_;
		size_t mf_, sf_;
	};

}