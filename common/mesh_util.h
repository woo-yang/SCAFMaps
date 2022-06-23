#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/triangle_triangle_adjacency.h>
#include "../common/mesh_type.h"

namespace common {
	int cal_cot_laplace(
		const Eigen::Matrix3Xd& V,
		const Eigen::Matrix3Xi& F,
		Eigen::SparseMatrix<double>& L);

	int triangulate_polygon(
		const Eigen::Matrix2Xd& V, //position of boundary verts
		const Eigen::Matrix2Xi& E, //id for boundarye edges
		const Eigen::Matrix2Xd& H, //position for verts of of inner boundary
		double max_area,  //maximum area constraint
		double min_angle, //minmial angle constraint
		Eigen::Matrix2Xd& NV, //position of output verts
		Eigen::Matrix3Xi& T); //id of output triangles

	int convert_uv_to_3d(
		const Eigen::Matrix2Xd& uv,
		Eigen::Matrix3Xd& V);

	void vertex_triangle_adjacency(
		const Eigen::Matrix3Xi& F, const int& n,
		std::vector<std::vector<int>>& v_f);

	void triangle_triangle_adjacency(
		const Eigen::Matrix3Xi& F,
		std::vector < std::vector<int>>& f_f);
}