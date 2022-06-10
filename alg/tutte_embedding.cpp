#include <sstream>
#include <algorithm>

#include <Eigen/Sparse>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/cat.h>

#include "tutte_embedding.h"
#include "../common/mesh_util.h"

namespace alg {

	int tutte_embedding(
		const common::tri_mesh& mesh,
		const Eigen::VectorXi& bnd,
		common::uv_mesh& uv_mesh)
	{

		Eigen::MatrixXd bound_uv;
		igl::map_vertices_to_circle(mesh.V_.transpose(), bnd, bound_uv);

		Eigen::SparseMatrix<double> L;
		common::cal_cot_laplace(mesh.V_, mesh.F_, L);

		for (auto i : bnd) {
			L.row(i) *= 0;L.coeffRef(i, i) = 1;
		}

		Eigen::MatrixXd B = Eigen::MatrixXd::Zero(mesh.V_.cols(),2);
		for (int i = 0; i < bnd.size(); ++i)
			B.row(bnd(i)) = 1 * bound_uv.row(i);

		Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>solver;
		solver.compute(L);
		Eigen::MatrixXd uv = solver.solve(B);

		uv_mesh.V_ = uv.transpose();
		uv_mesh.F_ = mesh.F_;

		return 1;
	}

    int triangulate_scaffold(
		const common::uv_mesh& uv_mesh,
		const Eigen::VectorXi& bnd,
		common::ext_mesh& extend_mesh)
	{
		int bnd_cnt = bnd.size();
		double avg_area = 0;
		for (int i = 0; i < uv_mesh.F_.cols(); ++i) {
			Eigen::Vector2d p1 = uv_mesh.V_.col(uv_mesh.F_(1, i)) - uv_mesh.V_.col(uv_mesh.F_(0, i));
			Eigen::Vector2d p2 = uv_mesh.V_.col(uv_mesh.F_(2, i)) - uv_mesh.V_.col(uv_mesh.F_(0, i));
			avg_area += (p1.x() * p2.y() - p2.x() * p1.y()) / 2.;
		}
		avg_area /= uv_mesh.F_.cols();

		Eigen::Vector2d hole = 
			uv_mesh.V_.col(uv_mesh.F_(0, 0)) +
			uv_mesh.V_.col(uv_mesh.F_(1, 0)) +
			uv_mesh.V_.col(uv_mesh.F_(2, 0));
		hole /= 3;

		Eigen::Vector2d uv_max = uv_mesh.V_.rowwise().maxCoeff();
		Eigen::Vector2d uv_min = uv_mesh.V_.rowwise().minCoeff();
		Eigen::Vector2d uv_mid = (uv_max + uv_min) / 2.;
		Eigen::Matrix2d ob;
		Eigen::Array2d scaf_range(3, 3);
		ob.row(0) = uv_mid.array() + scaf_range * ((uv_min - uv_mid).array());
		ob.row(1) = uv_mid.array() + scaf_range * ((uv_max - uv_mid).array());
		Eigen::Vector2d rect_len;
		rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);

		Eigen::Matrix2Xd V_bnd(2, bnd_cnt);
		for (int i = 0; i < bnd_cnt; ++i) {
			V_bnd.col(i) = uv_mesh.V_.col(bnd(i));
		}
		const int frame_point = 5;
		Eigen::Matrix2Xd V_frame(2, 4 * frame_point);
		for (int i = 0; i < frame_point; ++i) {
			V_frame.col(i) << ob(0, 0), ob(0, 1) + i * rect_len(1) / frame_point;
			V_frame.col(i + frame_point) << ob(0, 0) + i * rect_len(0) / frame_point, ob(1, 1);;
			V_frame.col(i + 2 * frame_point) << ob(1, 0), ob(1, 1)- i * rect_len(1) /frame_point;
			V_frame.col(i + 3 * frame_point) <<  ob(1, 0) - i * rect_len(0) / frame_point, ob(0, 1);
		}
		Eigen::Matrix2Xd V;
		igl::cat(2, V_bnd, V_frame, V);

		Eigen::Matrix2Xi E(2, V.cols());
		for (int i = 0; i < bnd_cnt + 4 * frame_point; i++) {
			E(0, i) = i;
			E(1, i) = i + 1;
		}
		E(1, bnd_cnt - 1) = 0;
		E(1, bnd_cnt + 4 * frame_point - 1) = bnd_cnt;

		common::uv_mesh scaffold_mesh;
		common::triangulate_polygon(V, E, hole, avg_area * 500, 30, scaffold_mesh.V_, scaffold_mesh.F_);

		//build ext mesh
		int inner_cnt = uv_mesh.V_.cols();
		int outer_cnt = scaffold_mesh.V_.cols() - bnd_cnt;

		Eigen::Matrix2Xd ext_V(2,inner_cnt + outer_cnt);
		ext_V.leftCols(inner_cnt) = uv_mesh.V_;
		ext_V.rightCols(outer_cnt) = scaffold_mesh.V_.rightCols(outer_cnt);

		Eigen::Matrix3Xi ext_F(3,uv_mesh.F_.cols() + scaffold_mesh.F_.cols());
		ext_F.leftCols(uv_mesh.F_.cols()) = uv_mesh.F_;
		Eigen::Matrix3Xi temp_F(3, scaffold_mesh.F_.cols());
		for (int i = 0; i < scaffold_mesh.F_.cols(); ++i) {
			for (int j = 0; j < 3; ++j) {
				if (scaffold_mesh.F_(j, i) < bnd_cnt)
					temp_F(j, i) = bnd[scaffold_mesh.F_(j, i)];
				else {
					temp_F(j, i) = scaffold_mesh.F_(j, i) + (inner_cnt - bnd_cnt);
				}
			}
		}
		ext_F.rightCols(scaffold_mesh.F_.cols()) = temp_F;
		extend_mesh.V_ = ext_V;
		extend_mesh.mf_ = uv_mesh.F_.cols();
		extend_mesh.sf_ = temp_F.cols();
		extend_mesh.F_ = ext_F;

		return 1;
    }
    
}