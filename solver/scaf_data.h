#pragma once 
#include <igl/MappingEnergyType.h>

#include <Eigen/Sparse>

struct scaf_data
{
	long mv_num, mf_num;
	long sv_num, sf_num;
	long v_num , f_num;

	Eigen::MatrixXd V_m; 
	Eigen::MatrixXi F_m; 

	Eigen::MatrixXd uv_w, uv_o;
	Eigen::MatrixXi F_w, F_s;

	Eigen::VectorXd M_m, M_s;
	Eigen::MatrixXd Ri_m, Ji_m, Ri_s, Ji_s;
	Eigen::MatrixXd W_m, W_s;

	Eigen::SparseMatrix<double> Dx_m, Dy_m; //F*V
	Eigen::SparseMatrix<double> Dx_s, Dy_s;

	double energy;
	double mesh_area;

	Eigen::VectorXi b;
	Eigen::MatrixXd bc;
	Eigen::VectorXi inner_bnd;
	Eigen::VectorXi frame_ids;
	const double soft_const_p = 1e4;

	const int dim = 2;
	bool has_pre_calc = false;
	igl::MappingEnergyType energy_type = igl::MappingEnergyType::SYMMETRIC_DIRICHLET;
};