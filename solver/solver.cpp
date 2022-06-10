#include "solver.h"
#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/slim.h>
#include <Eigen/Sparse>

namespace slim {

	void pre_calc(scaf_data& d);
	void compute_jacobians(scaf_data& d, const Eigen::MatrixXd& uv_new, bool whole = true);
	double compute_energy(scaf_data& d, const Eigen::MatrixXd& uv_new, bool whole = true);
	void solve_weighted_arap(scaf_data& d, Eigen::MatrixXd& uv_o);

	void compute_surface_gradient_matrix(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::SparseMatrix<double>& D1,
		Eigen::SparseMatrix<double>& D2);

	void compute_scaffold_gradient_matrix(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::VectorXi& inner_bnd,
		Eigen::SparseMatrix<double>& D1,
		Eigen::SparseMatrix<double>& D2);

	void build_surface_linear_system(
		scaf_data& d,
		Eigen::SparseMatrix<double>& L,
		Eigen::VectorXd& rhs);

	void build_scaffold_linear_system(
		scaf_data& d,
		const Eigen::VectorXi& known_ids,
		const Eigen::VectorXi& unknown_ids,
		const Eigen::VectorXd& known_pos,
		Eigen::SparseMatrix<double>& L,
		Eigen::VectorXd& rhs);

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
		scaf_data& d)
	{
		d.V_m = V_m.transpose(); d.F_m = F_m.transpose();
		d.uv_w = uv_w.transpose(); d.uv_o = uv_w.transpose();
		d.F_w = F_w.transpose(); d.F_s = F_s.transpose();
		d.b = b; d.bc = bc.transpose(); 
		d.inner_bnd = inner_bnd; d.frame_ids = frame_ids;
		std::sort(d.frame_ids.begin(), d.frame_ids.end());

		d.v_num = d.uv_w.rows(); d.f_num = d.F_w.rows();
		d.mv_num = d.V_m.rows(); d.mf_num = d.F_m.rows();
		d.sv_num = d.v_num - d.mv_num; d.sf_num = d.F_s.rows();

		igl::doublearea(d.V_m, d.F_m, d.M_m);	
		igl::doublearea(d.uv_w, d.F_s, d.M_s);
		d.M_m /= 2.; d.M_s /= 2.;
		d.mesh_area = d.M_m.sum() + d.M_s.sum();
	
		slim::pre_calc(d);
		d.energy = compute_energy(d, d.uv_o) / d.mesh_area;

	}
	void pre_calc(scaf_data& d) 
	{
		if (d.has_pre_calc)return;

		compute_surface_gradient_matrix(d.V_m, d.F_m, d.Dx_m, d.Dy_m);
		compute_scaffold_gradient_matrix(d.uv_w, d.F_s, d.inner_bnd, d.Dx_s, d.Dy_s);

		int dim = d.dim;
		d.Dx_m.makeCompressed();
		d.Dy_m.makeCompressed();
		d.Ri_m = Eigen::MatrixXd::Zero(d.mf_num, dim * dim);
		d.Ji_m.resize(d.mf_num, dim * dim);
		d.W_m.resize(d.mf_num, dim * dim);

		d.Dx_s.makeCompressed();
		d.Dy_s.makeCompressed();
		d.Ri_s = Eigen::MatrixXd::Zero(d.sf_num, dim * dim);
		d.Ji_s.resize(d.sf_num, dim * dim);
		d.W_s.resize(d.sf_num, dim * dim);

		d.has_pre_calc = true;
	}

	void compute_jacobians(
		const Eigen::MatrixXd& uv,
		const Eigen::SparseMatrix<double>& Dx,
		const Eigen::SparseMatrix<double>& Dy,
		Eigen::MatrixXd& Ji)
	{
		// Ji=[D1*u,D2*u,D1*v,D2*v];
		Ji.resize(Dx.rows(), 4);
		Ji.col(0) = Dx * uv.col(0);
		Ji.col(1) = Dy * uv.col(0);
		Ji.col(2) = Dx * uv.col(1);
		Ji.col(3) = Dy * uv.col(1);
	}

	void compute_jacobians(scaf_data& d, const Eigen::MatrixXd& uv_new, bool whole)
	{
		compute_jacobians(uv_new.topRows(d.mv_num), d.Dx_m, d.Dy_m, d.Ji_m);
		if (whole) compute_jacobians(uv_new, d.Dx_s, d.Dy_s, d.Ji_s);
	}

	double compute_soft_constraint_energy(scaf_data& d, const Eigen::MatrixXd& uv)
	{
		double e = 0;
		for (int i = 0; i < d.b.rows(); i++)
			e += d.soft_const_p * (d.bc.row(i) - uv.row(d.b(i))).squaredNorm();
		return e;
	}

	double compute_energy(scaf_data& d, const Eigen::MatrixXd& uv_new, bool whole)
	{
		compute_jacobians(d, uv_new, whole);
		double energy = 0;
		energy += igl::mapping_energy_with_jacobians(d.Ji_m, d.M_m, d.energy_type, 0);
		if (whole)
			energy += igl::mapping_energy_with_jacobians(d.Ji_s, d.M_s, d.energy_type, 0);

		energy += compute_soft_constraint_energy(d, uv_new);

		return energy;
	}

	void compute_surface_gradient_matrix(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::SparseMatrix<double>& D1,
		Eigen::SparseMatrix<double>& D2)
	{
		Eigen::MatrixXd F1, F2, F3;
		igl::local_basis(V, F, F1, F2, F3);
		Eigen::SparseMatrix<double> G;
		igl::grad(V, F, G);
		Eigen::SparseMatrix<double> Dx = G.block(0, 0, F.rows(), V.rows());
		Eigen::SparseMatrix<double> Dy = G.block(F.rows(), 0, F.rows(), V.rows());
		Eigen::SparseMatrix<double> Dz = G.block(2 * F.rows(), 0, F.rows(), V.rows());

		D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
		D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;

	}

	template<typename DerivedV, typename DerivedF>
	inline void adjusted_grad(
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::SparseMatrix<typename DerivedV::Scalar>& G,
		double eps) 
	{
		Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>
			eperp21(F.rows(), 3), eperp13(F.rows(), 3);
		int fixed = 0;
		for (int i = 0; i < F.rows(); ++i) {
			// renaming indices of vertices of triangles for convenience
			int i1 = F(i, 0);
			int i2 = F(i, 1);
			int i3 = F(i, 2);

			// #F x 3 matrices of triangle edge vectors, named after opposite vertices
			Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v32 = V.row(i3) - V.row(i2);
			Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v13 = V.row(i1) - V.row(i3);
			Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v21 = V.row(i2) - V.row(i1);
			Eigen::Matrix<typename DerivedV::Scalar, 1, 3> n = v32.cross(v13);
			// area of parallelogram is twice area of triangle
			// area of parallelogram is || v1 x v2 ||
			// This does correct l2 norm of rows, so that it contains #F list of twice
			// triangle areas
			double dblA = std::sqrt(n.dot(n));
			Eigen::Matrix<typename DerivedV::Scalar, 1, 3> u(0, 0, 1);
			if (dblA > eps) {
				// now normalize normals to get unit normals
				u = n / dblA;
			}
			else {
				// Abstract equilateral triangle v1=(0,0), v2=(h,0), v3=(h/2, (sqrt(3)/2)*h)
				fixed++;
				// get h (by the area of the triangle)
				dblA = eps;
				double h = sqrt((dblA) / sin(
					M_PI / 3.0)); // (h^2*sin(60))/2. = Area => h = sqrt(2*Area/sin_60)
				
				Eigen::Vector3d v1, v2, v3;
				v1 << 0, 0, 0;
				v2 << h, 0, 0;
				v3 << h / 2., (sqrt(3) / 2.)* h, 0;

				// now fix v32,v13,v21 and the normal
				v32 = v3 - v2;
				v13 = v1 - v3;
				v21 = v2 - v1;
				n = v32.cross(v13);
			}

			// rotate each vector 90 degrees around normal
			double norm21 = std::sqrt(v21.dot(v21));
			double norm13 = std::sqrt(v13.dot(v13));
			eperp21.row(i) = u.cross(v21);
			eperp21.row(i) =
				eperp21.row(i) / std::sqrt(eperp21.row(i).dot(eperp21.row(i)));
			eperp21.row(i) *= norm21 / dblA;
			eperp13.row(i) = u.cross(v13);
			eperp13.row(i) =
				eperp13.row(i) / std::sqrt(eperp13.row(i).dot(eperp13.row(i)));
			eperp13.row(i) *= norm13 / dblA;
		}

		std::vector<int> rs;
		rs.reserve(F.rows() * 4 * 3);
		std::vector<int> cs;
		cs.reserve(F.rows() * 4 * 3);
		std::vector<double> vs;
		vs.reserve(F.rows() * 4 * 3);

		// row indices
		for (int r = 0; r < 3; r++) {
			for (int j = 0; j < 4; j++) {
				for (int i = r * F.rows(); i < (r + 1) * F.rows(); i++) rs.push_back(i);
			}
		}

		// column indices
		for (int r = 0; r < 3; r++) {
			for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 1));
			for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 0));
			for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 2));
			for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 0));
		}

		// values
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 0));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 0));
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 0));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 0));
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 1));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 1));
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 1));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 1));
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 2));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 2));
		for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 2));
		for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 2));

		// create sparse gradient operator matrix
		G.resize(3 * F.rows(), V.rows());
		std::vector<Eigen::Triplet<typename DerivedV::Scalar> > triplets;
		for (int i = 0; i < (int)vs.size(); ++i) {
			triplets.push_back(Eigen::Triplet<typename DerivedV::Scalar>(rs[i],cs[i],vs[i]));
		}
		G.setFromTriplets(triplets.begin(), triplets.end());
		//  std::cout<<"Adjusted"<<fixed<<std::endl;
	};

	void compute_scaffold_gradient_matrix(
		const Eigen::MatrixXd& UV,
		const Eigen::MatrixXi& F_s,
		const Eigen::VectorXi& inner_bnd,
		Eigen::SparseMatrix<double>& D1,
		Eigen::SparseMatrix<double>& D2)
	{
		Eigen::SparseMatrix<double> G;
		int vn = UV.rows();
		Eigen::MatrixXd V = Eigen::MatrixXd::Zero(vn, 3);
		V.leftCols(2) = UV;

		double min_bnd_edge_len = INFINITY;
		int acc_bnd = 0;

		int current_size = inner_bnd.size();

		for (int e = acc_bnd; e < acc_bnd + current_size - 1; e++)
		{
			min_bnd_edge_len = (std::min)(min_bnd_edge_len,(UV.row(inner_bnd(e)) -UV.row(inner_bnd(e + 1))).squaredNorm());
		}
		min_bnd_edge_len = (std::min)(min_bnd_edge_len,(UV.row(inner_bnd(acc_bnd)) -UV.row(inner_bnd(acc_bnd + current_size -1))).squaredNorm());
		acc_bnd += current_size;
		double area_threshold = min_bnd_edge_len / 4.0;
		adjusted_grad(V, F_s, G, area_threshold);
		Eigen::SparseMatrix<double> Dx = G.block(0, 0, F_s.rows(), vn);
		Eigen::SparseMatrix<double> Dy = G.block(F_s.rows(), 0, F_s.rows(), vn);
		Eigen::SparseMatrix<double> Dz = G.block(2 * F_s.rows(), 0, F_s.rows(), vn);

		Eigen::MatrixXd F1, F2, F3;
		igl::local_basis(V, F_s, F1, F2, F3);
		D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy +F1.col(2).asDiagonal() * Dz;
		D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy +F2.col(2).asDiagonal() * Dz;
	}

	Eigen::MatrixXd slim_solve(scaf_data& d)
	{
		compute_jacobians(d, d.uv_o, true);
		igl::slim_update_weights_and_closest_rotations_with_jacobians(d.Ji_m, d.energy_type, 0, d.W_m, d.Ri_m);
		igl::slim_update_weights_and_closest_rotations_with_jacobians(d.Ji_s, d.energy_type, 0, d.W_s, d.Ri_s);
		
		Eigen::MatrixXd dest_res;
		solve_weighted_arap(d, dest_res);

		std::function<double(Eigen::MatrixXd&)> energy_func = [&](Eigen::MatrixXd& uv)
		{ return compute_energy(d, uv); };

		d.energy = igl::flip_avoiding_line_search(d.F_w, d.uv_o, dest_res, energy_func) / d.mesh_area;
		return d.uv_o;
	}

	void solve_weighted_arap(
		scaf_data& d,
		Eigen::MatrixXd& dest_res)
	{
		int dim = d.dim;
		long v_n = d.v_num;

		const Eigen::VectorXi& bnd_ids = d.frame_ids;
		const auto bnd_n = bnd_ids.size();

		Eigen::MatrixXd bnd_pos;
		igl::slice(d.uv_w, bnd_ids, 1, bnd_pos);

		Eigen::VectorXi known_ids(bnd_n * dim);
		Eigen::VectorXi unknown_ids((v_n - bnd_n) * dim);

		{ // get the complement of bnd_ids.
			int assign = 0, i = 0;
			for (int get = 0; i < v_n && get < bnd_ids.size(); i++)
			{
				if (bnd_ids(get) == i)
					get++;
				else
					unknown_ids(assign++) = i;
			}
			while (i < v_n)
				unknown_ids(assign++) = i++;
			assert(assign + bnd_ids.size() == v_n);
		}

		Eigen::VectorXd known_pos(bnd_n * dim);
		for (int d = 0; d < dim; d++)
		{
			known_ids.segment(d * bnd_n, bnd_n) = bnd_ids.array() + d * v_n;
			known_pos.segment(d * bnd_n, bnd_n) = bnd_pos.col(d);
			unknown_ids.segment(d * (v_n - bnd_n), v_n - bnd_n) = 
				unknown_ids.topRows(v_n - bnd_n).array() + d * v_n;
		}

		Eigen::SparseMatrix<double> L;
		Eigen::VectorXd rhs;

		// fixed frame solving:
		// x_e as the fixed frame, x_u for unknowns (mesh + unknown scaffold)
		// min ||(A_u*x_u + A_e*x_e) - b||^2
		// => A_u'*A_u*x_u + A_u'*A_e*x_e = Au'*b
		// => A_u'*A_u*x_u  = Au'* (b - A_e*x_e) := Au'* b_u
		// => L * x_u = rhs
		//
		// separate matrix build:
		// min ||A_m x_m - b_m||^2 + ||A_s x_all - b_s||^2 + soft + proximal
		// First change dimension of A_m to fit for x_all
		// (Not just at the end, since x_all is flattened along dimensions)
		// L = A_m'*A_m + A_s'*A_s + soft + proximal
		// rhs = A_m'* b_m + A_s' * b_s + soft + proximal
		//
		using namespace std;
		Eigen::SparseMatrix<double> L_m, L_s;
		Eigen::VectorXd rhs_m, rhs_s;
		build_surface_linear_system(d, L_m, rhs_m); 
		build_scaffold_linear_system(d, known_ids, unknown_ids, known_pos, L_s, rhs_s);;

		L = L_m + L_s;
		rhs = rhs_m + rhs_s;
		L.makeCompressed();

		Eigen::VectorXd unknown_Uc((v_n - bnd_n) * dim), Uc(dim * v_n);

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		solver.analyzePattern(L);
		solver.factorize(L);
		if (solver.info() != Eigen::Success) {
			// decomposition failed
			return;
		}
		unknown_Uc = solver.solve(rhs);

		igl::slice_into(unknown_Uc, unknown_ids.matrix(), 1, Uc);
		igl::slice_into(known_pos, known_ids.matrix(), 1, Uc);

		dest_res = Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(Uc.data(), v_n, dim);

	}

	void buildAm(const Eigen::VectorXd& sqrt_M,
		const Eigen::SparseMatrix<double>& Dx,
		const Eigen::SparseMatrix<double>& Dy,
		const Eigen::MatrixXd& W,
		Eigen::SparseMatrix<double>& Am)
	{
		std::vector<Eigen::Triplet<double>> IJV;

		Eigen::SparseMatrix<double> MDx = sqrt_M.asDiagonal() * Dx;
		Eigen::SparseMatrix<double> MDy = sqrt_M.asDiagonal() * Dy;
		Eigen::SparseMatrix<double> MDz;
		igl::slim_buildA(MDx, MDy, MDz, W, IJV);

		Am.setFromTriplets(IJV.begin(), IJV.end());
		Am.makeCompressed();
	}

	void buildRhs(
		const Eigen::VectorXd& sqrt_M,
		const Eigen::MatrixXd& W,
		const Eigen::MatrixXd& Ri,
		Eigen::VectorXd& f_rhs)
	{
		const int dim = 2;
		const int f_n = W.rows();
		f_rhs.resize(dim * dim * f_n);

		/*b = [W11*R11 + W12*R21; (formula (36))
				W11*R12 + W12*R22;
				W21*R11 + W22*R21;
				W21*R12 + W22*R22];*/
		for (int i = 0; i < f_n; i++)
		{
			auto sqrt_area = sqrt_M(i);
			f_rhs(i + 0 * f_n) = sqrt_area * (W(i, 0) * Ri(i, 0) + W(i, 1) * Ri(i, 1));
			f_rhs(i + 1 * f_n) = sqrt_area * (W(i, 0) * Ri(i, 2) + W(i, 1) * Ri(i, 3));
			f_rhs(i + 2 * f_n) = sqrt_area * (W(i, 2) * Ri(i, 0) + W(i, 3) * Ri(i, 1));
			f_rhs(i + 3 * f_n) = sqrt_area * (W(i, 2) * Ri(i, 2) + W(i, 3) * Ri(i, 3));
		}	
	}

	void add_soft_constraints(
		scaf_data& s, 
		Eigen::SparseMatrix<double>& L,
		Eigen::VectorXd& rhs)
	{
		int v_n = s.v_num - s.frame_ids.size();
		for (int d = 0; d < s.dim; d++)
		{
			for (int i = 0; i < s.b.rows(); i++)
			{
				int v_idx = s.b(i);
				rhs(d * v_n + v_idx) += s.soft_const_p * s.bc(i, d); // rhs
				L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.soft_const_p; // diagonal of matrix
				
			}
		}
	}
	void build_surface_linear_system(
		scaf_data& d,
		Eigen::SparseMatrix<double>& L,
		Eigen::VectorXd& rhs)
	{
		const int v_n = d.v_num - (d.frame_ids.size());
		const int dim = d.dim;
		const int f_n = d.mf_num;

		// to get the  complete A
		Eigen::VectorXd sqrtM = d.M_m.array().sqrt();
		Eigen::SparseMatrix<double> A(dim * dim * f_n, dim * v_n);

		auto decoy_Dx_m = d.Dx_m; decoy_Dx_m.conservativeResize(d.W_m.rows(), v_n);
		auto decoy_Dy_m = d.Dy_m; decoy_Dy_m.conservativeResize(d.W_m.rows(), v_n);

		buildAm(sqrtM, decoy_Dx_m, decoy_Dy_m, d.W_m, A);

		Eigen::SparseMatrix<double> At = A.transpose();
		At.makeCompressed();

		L = At * A;

		Eigen::VectorXd frhs;
		buildRhs(sqrtM, d.W_m, d.Ri_m, frhs);
		rhs = At * frhs;

		add_soft_constraints(d, L, rhs);
	}

	void build_scaffold_linear_system(
		scaf_data& d,
		const Eigen::VectorXi& known_ids,
		const Eigen::VectorXi& unknown_ids,
		const Eigen::VectorXd& known_pos,
		Eigen::SparseMatrix<double>& L,
		Eigen::VectorXd& rhs)
	{
		const int f_n = d.W_s.rows();
		const int v_n = d.Dx_s.cols();
		const int dim = d.dim;

		Eigen::VectorXd sqrt_M = d.M_s.array().sqrt();
		Eigen::SparseMatrix<double> A(dim * dim * f_n, dim * v_n);
		buildAm(sqrt_M, d.Dx_s, d.Dy_s, d.W_s, A);

		// 'manual slicing for A(:, unknown/known)'
		Eigen::SparseMatrix<double> Au, Ae;
		{
			int xm = A.rows(), xn = A.cols();
			int ym = xm;
			int yn = unknown_ids.size();
			int ykn = known_ids.size();

			std::vector<int> CI(xn, -1), CKI(xn, -1);

			for (int i = 0; i < yn; i++)CI[unknown_ids(i)] = i;
			for (int i = 0; i < ykn; i++)CKI[known_ids(i)] = i;
			
			Eigen::SparseMatrix<double, Eigen::ColMajor> dyn_Y(ym, yn), dyn_K(ym, ykn);
			// Take a guess at the number of nonzeros (this assumes uniform distribution
			// not banded or heavily diagonal)
			dyn_Y.reserve(A.nonZeros());
			dyn_K.reserve(A.nonZeros() * ykn / xn);
			// Iterate over outside
			for (int k = 0; k < A.outerSize(); ++k)
			{
				// Iterate over inside
				if (CI[k] != -1) {
					for (typename Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it){
						dyn_Y.coeffRef(it.row(), CI[it.col()]) = it.value();
					}
				}
				else {
					for (typename Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it){
						dyn_K.coeffRef(it.row(), CKI[it.col()]) = it.value();
					}
				}
			}
			Au = Eigen::SparseMatrix<double>(dyn_Y);
			Ae = Eigen::SparseMatrix<double>(dyn_K);
		}

		Eigen::SparseMatrix<double> Aut = Au.transpose();
		Aut.makeCompressed();

		L = Aut * Au;

		Eigen::VectorXd frhs;
		buildRhs(sqrt_M, d.W_s, d.Ri_s, frhs);

		rhs = Aut * (frhs - Ae * known_pos);
	}

	void slim_update(
		const Eigen::MatrixXd& uv_w,
		const Eigen::MatrixXi& F_w,
		const Eigen::MatrixXi& F_s,
		const Eigen::VectorXi& frame_ids,
		scaf_data& d)
	{
		d.uv_w = uv_w.transpose(); d.uv_o = uv_w.transpose();
		d.F_w = F_w.transpose(); d.F_s = F_s.transpose();
		d.frame_ids = frame_ids;
		std::sort(d.frame_ids.begin(), d.frame_ids.end());

		d.v_num = d.uv_w.rows(); d.f_num = d.F_w.rows();
		d.sv_num = d.v_num - d.mv_num; d.sf_num = d.F_s.rows();

		igl::doublearea(d.uv_w, d.F_s, d.M_s);
		d.M_s /= 2.;
		d.mesh_area = d.M_m.sum() + d.M_s.sum();

		compute_scaffold_gradient_matrix(d.uv_w, d.F_s, d.inner_bnd, d.Dx_s, d.Dy_s);
		d.Dx_s.makeCompressed();
		d.Dy_s.makeCompressed();
		int dim = d.dim;
		d.Ri_s = Eigen::MatrixXd::Zero(d.sf_num, dim * dim);
		d.Ji_s.resize(d.sf_num, dim * dim);
		d.W_s.resize(d.sf_num, dim * dim);

		d.energy = compute_energy(d, d.uv_o) / d.mesh_area;
	}
}