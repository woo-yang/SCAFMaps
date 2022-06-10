#pragma once
#include <vector>

#include <igl/boundary_loop.h>
#include <igl/Timer.h>

#include "mesh_optimization.h"
#include "tutte_embedding.h"
#include "../solver/solver.h"
#include "../solver/scaf_data.h"


namespace alg {
	int mesh_optimization(
		const common::tri_mesh& mesh,
		common::ext_mesh& extend_mesh,
		int max_iter, 
		std::function<void(const common::ext_mesh& V)> call_back)
	{
		Eigen::VectorXi inner_bnd, outer_bnd;
		igl::boundary_loop(extend_mesh.F_.leftCols(extend_mesh.mf_).transpose(), inner_bnd);
		igl::boundary_loop(extend_mesh.F_.transpose(), outer_bnd);

		scaf_data data;
		slim::slim_precompute(
			mesh.V_, mesh.F_, extend_mesh.V_,
			extend_mesh.F_, extend_mesh.F_.rightCols(extend_mesh.sf_),
			inner_bnd, outer_bnd, Eigen::VectorXi(), Eigen::MatrixXd(), data);

		igl::Timer timer;
		for (int k = 0; k < max_iter; ++k) {
			timer.start();
			//solve symmetric dirichlet energy
			
			extend_mesh.V_ = slim::slim_solve(data).transpose();
			std::cout << "iteration "<< k 
				<< " ,time = " << timer.getElapsedTime()
				<< " ,energy = " << data.energy << std::endl;
	
			//build uv_mesh
			common::uv_mesh uv_mesh;
			uv_mesh.V_ = extend_mesh.V_.leftCols(mesh.V_.cols());
			uv_mesh.F_ = extend_mesh.F_.leftCols(extend_mesh.mf_);

			//regenerate scaffold
			alg::triangulate_scaffold(uv_mesh, inner_bnd, extend_mesh);

			igl::boundary_loop(extend_mesh.F_.transpose(), outer_bnd);
			slim::slim_update(extend_mesh.V_, extend_mesh.F_,
				extend_mesh.F_.rightCols(extend_mesh.sf_), outer_bnd, data);

			if (call_back)call_back(extend_mesh);
		}
		return 1;
	}
}