#pragma once
#include "scaf_data.h"

#include <Eigen/Dense>

namespace slim {

	void slim_precompute(
		const Eigen::MatrixXd& V_m,
		const Eigen::MatrixXi& F_m,
		const Eigen::MatrixXd& uv_w,
		const Eigen::MatrixXi& F_w,
		const Eigen::MatrixXi& F_s,
		const Eigen::VectorXi& inner_bnd,
		const Eigen::VectorXi& frame_ids,
		const Eigen::VectorXi& b,
		const Eigen::MatrixXd& bc,
		scaf_data& d);

	void slim_update(
		const Eigen::MatrixXd& uv_w,
		const Eigen::MatrixXi& F_w,
		const Eigen::MatrixXi& F_s,
		const Eigen::VectorXi& frame_ids,
		scaf_data& d);

	Eigen::MatrixXd slim_solve(scaf_data& d);

}