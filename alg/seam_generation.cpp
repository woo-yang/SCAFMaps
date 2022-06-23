#include "seam_generation.h"

#include "../common/utilities.h"
#include "../common/mesh_util.h"
#include "../io/obj_io.h"

#include <vector>
#include <algorithm>
#include <queue>

#include <igl/dijkstra.h>
#include <igl/boundary_loop.h>
#include <igl/adjacency_list.h>
#include <igl/mat_max.h>
#include <igl/unique.h>
#include <igl/boundary_facets.h>

#include <math.h>

namespace alg {

	int calc_visibility(
		const common::tri_mesh& mesh,
		Eigen::VectorXd& Vn)
	{
		Vn.setConstant(mesh.V_.cols(), 1);
		return 1;
	}

	int calc_region_radius(
		const common::tri_mesh& mesh,
		Eigen::MatrixXd& Rn,int r)
	{
		//For r >0 we define the region radius R as the product
		//of rand the length of the longest edge emanating from n
		Rn.resize(mesh.V_.cols(), r + 1);

		std::vector<double> max_edge(mesh.V_.cols(), -1);
		for (int i = 0; i < mesh.F_.cols(); ++i) {
			for (int j = 0; j < 3; ++j) {
				int p = mesh.F_(j, i), q = mesh.F_((j + 1) % 3, i);
				double len = (mesh.V_.col(p) - mesh.V_.col(q)).norm();
				if (len > max_edge[p])max_edge[p] = len;
				if (len > max_edge[q])max_edge[q] = len;
			}
		}

		for (int i = 0; i < mesh.V_.cols(); ++i) {
			for (int k = 0; k <= r; ++k)
				Rn(i, k) = k * max_edge[i];
		}
		return 1;
	}

	double  calc_distortion_with_face(
		const common::tri_mesh& mesh,
		const int n ,const std::vector<int>& face)
	{
		Eigen::Vector3d p0 = mesh.V_.col(n);
		Eigen::MatrixXi F;
		std::vector<int> temp = face;
		Eigen::Map<Eigen::VectorXi> idx(temp.data(), temp.size());
		igl::slice(mesh.F_, idx, 2, F);
		Eigen::VectorXi bnd;
		igl::boundary_loop(F.transpose(), bnd);
		double angles = 0;
		for (int j = 0; j < bnd.size(); ++j) {
			Eigen::Vector3d p1 = mesh.V_.col(bnd[j]);
			Eigen::Vector3d p2 = mesh.V_.col(bnd[(j + 1) % bnd.size()]);
			angles += std::acos((p1 - p0).normalized().dot((p2 - p0).normalized()));
		}
		return 2 * M_PI - angles*(M_PI/180) / 2 * M_PI;
	}

	double calc_distortion(
		const common::tri_mesh& mesh,
		Eigen::VectorXd& Dn, int r)
	{
		Eigen::MatrixXd Rn; //V*(r+1) 0,1,2,...r
		calc_region_radius(mesh, Rn, r); 

		std::vector<std::vector<int>> v_f;
		common::vertex_triangle_adjacency(mesh.F_, mesh.V_.cols(), v_f);

		std::vector < std::vector<int>> f_f(mesh.F_.cols());
		common::triangle_triangle_adjacency(mesh.F_, f_f);

		Eigen::Matrix3Xd f_center(3, mesh.F_.cols());
		for (int i = 0; i < mesh.F_.cols(); ++i) {
			f_center.col(i) = (mesh.V_.col(mesh.F_(0, i)) +
				mesh.V_.col(mesh.F_(1, i)) +
				mesh.V_.col(mesh.F_(2, i))) / 3;
		}

		Eigen::MatrixXd Dr(mesh.V_.cols(), r + 1);

		for (int n = 0; n < mesh.V_.cols(); ++n) {
			auto p0 = mesh.V_.col(n);
			std::queue<int> f_queue;
			std::vector<bool> f_visit(mesh.F_.cols(), false);
			std::vector<std::vector<int>> f_r(r + 1);
			//For r =0, the region contains the triangles adjacent to n.
			f_r[0] = v_f[n];

			for (auto f : v_f[n]) {
				if ((f_center.col(f) - p0).norm() < Rn(n, 1)) {
					f_queue.push(f); f_visit[f] = true;				
					for (int i = 1; i < r + 1; ++i)f_r[i].push_back(f);
				}			
			}

			while (!f_queue.empty()) {
				int cur_f = f_queue.front();
				f_queue.pop();
				for (auto f : f_f[cur_f]) {
					if (f_visit[f]) continue;
					for (int k = 1; k <= r; ++k) {
						if ((f_center.col(f) - p0).norm() < Rn(n, k)) {
							f_queue.push(f); f_visit[f] = true;
							for (int i = k; i < r + 1; ++i)f_r[i].push_back(f);
							break;
						}
					}
				}
			}
			for (int k = 0; k <= r; ++k) {			
				Dr(n, k) = calc_distortion_with_face(mesh, n, f_r[k]);
			}
		}

		//If the sum of angles around a boundary vertex n is less than 2pi, 
		//then the sub - mesh around it is locally developable and D(n) = 0.
		Eigen::MatrixXi  bnd;
		igl::boundary_facets(mesh.F_.transpose(), bnd);
		std::for_each(bnd.data(), bnd.data() + bnd.size(),
			[&](int v) {Dr.row(v).setZero(); });

		Eigen::VectorXi max_r;
		igl::mat_max(Dr, 2, Dn, max_r);

		//overlap avoidance
		for (int n = 0; n < mesh.V_.cols(); ++n) {
			for (int m = 0; m < mesh.V_.cols(); ++m) {
				if (n == m)continue;
				if (Rn(m, max_r(m)) > (mesh.V_.col(n) - mesh.V_.col(m)).norm()) continue;			
				if ((Dn(n) - Dn(m)) < 1e-3 && Dr(n, 0) > Dr(m, 0)) {
					Dn(m) = Dr(m, 0);
				}
				else if (Dn(n) > Dn(m)) {
					Dn(m) = Dr(m, 0);
				}
			}
		}
		return std::accumulate(Dn.begin(), Dn.end(), 0);
	}

	int select_terminal_vertices(
		const common::tri_mesh& mesh,
		int r,double percentage,
		std::vector<int>& terminal_v)
	{
		Eigen::VectorXd Dn, Vn;
		calc_visibility(mesh, Vn);
		double D_M = calc_distortion(mesh, Dn, r);

		std::vector<int> order;
		for (int i = 0; i < Dn.size(); ++i)
			order.push_back(i);
		
		std::sort(order.begin(), order.end(),[&](int l, int r) {
			return Dn(l) / Vn(l) > Dn(r) / Vn(r);
			});

		double s = 0;

		//The selection guarantees that the distortion 
		//after the seam addition will be less than D.The
		int i = 0;
		while (s < percentage * D_M) {
			terminal_v.push_back(order[i]);
			s += Dn(order[i]);
			i++;
		}

		//if the surface a priori contains one or more boundary loops,
		//we add all the boundary vertices to T.
		Eigen::MatrixXi  bnd;
		igl::boundary_facets(mesh.F_.transpose(), bnd);
		std::unique_copy(bnd.data(), bnd.data() + bnd.size(), std::back_inserter(terminal_v));

		return 1;
	}
	
	double kruskal_minimal_spanning_tree(
		const Eigen::MatrixXd& adjacency_matrix,
		std::set<std::pair<int,int>>& mst)
	{
		int mst_n = adjacency_matrix.cols() - 1;
		using edge = std::pair<int, int>;
		std::vector<std::pair<edge, double>> edge_set;
		
		for (int i = 0; i < mst_n; ++i) {
			for (int j = i + 1; j < mst_n; ++j) {
				edge_set.push_back({ { i,j }, adjacency_matrix(i, j) });
			}
		}
		std::sort(edge_set.begin(), edge_set.end(), 
			[](const std::pair<edge,double>& l, const std::pair<edge, double>& r) {
				return l.second < r.second;
			});
		common::UnionFind uf(mst_n + 1);
		int e_n = 0;
		double price = 0;

		for (const auto& it : edge_set) {
			auto e = it.first;
			int x = e.first, y = e.second;
			if (uf.find(x) == uf.find(y)) continue;
	
			uf.merge(x, y);
			mst.insert(e);
			e_n++;
			price += it.second;
			if (e_n == mst_n)break;
		}

		return price;

	}
	int dijkstra(
		const Eigen::MatrixXd V,
		const Eigen::MatrixXi F,
		const std::vector<std::vector<int> >& VV,
		const int& source,const std::set<int>& targets,
		Eigen::VectorXd& min_distance,
		Eigen::VectorXi& previous)
	{
		int numV = VV.size();
		Eigen::MatrixXi  bnd;
		igl::boundary_facets(F, bnd);
		Eigen::MatrixXi bnd_flag = Eigen::MatrixXi::Constant(numV, numV, 0);
		for (int i = 0; i < bnd.rows(); ++i) {
			bnd_flag(bnd(i, 0), bnd(i, 1)) =
			bnd_flag(bnd(i, 1), bnd(i, 0)) = 1;
		}

		min_distance.setConstant(numV, 1, std::numeric_limits<double>::infinity());
		min_distance[source] = 0;
		previous.setConstant(numV, 1, -1);
		std::set<std::pair<double, int> > vertex_queue;
		vertex_queue.insert(std::make_pair(min_distance[source], source));

		while (!vertex_queue.empty())
		{
			double dist = vertex_queue.begin()->first;
			int u = vertex_queue.begin()->second;
			vertex_queue.erase(vertex_queue.begin());

			if (targets.find(u) != targets.end())
				return u;

			// Visit each edge exiting u
			const std::vector<int>& neighbors = VV[u];
			for (auto neighbor_iter = neighbors.begin();
				neighbor_iter != neighbors.end();neighbor_iter++)
			{
				int v = *neighbor_iter;
				double weight_uv = 0;
				if (bnd_flag(u, v)) weight_uv = 1e-8;
				else weight_uv = (V.row(u) - V.row(v)).norm();
				double distance_through_u = dist + weight_uv;
				if (distance_through_u < min_distance[v]) {
					vertex_queue.erase(std::make_pair(min_distance[v], v));

					min_distance[v] = distance_through_u;
					previous[v] = u;
					vertex_queue.insert(std::make_pair(min_distance[v], v));

				}

			}
		}
		return -1;
	}
	
	int gen_minimal_steiner_tree(
		const common::tri_mesh& mesh,
		const std::vector<int>& terminal_v,
		std::set<std::pair<int, int>>& edge_set)
	{
		std::set<int> t_v;
		std::unique_copy(terminal_v.begin(), terminal_v.end(), std::inserter(t_v, t_v.end()));

		std::vector<std::vector<int>> vv;
		igl::adjacency_list(mesh.F_.transpose(), vv);

		Eigen::MatrixXd distance(mesh.V_.cols(), mesh.V_.cols());
		Eigen::MatrixXi previous(mesh.V_.cols(), mesh.V_.cols());
		std::set<int> target = { static_cast<int>(mesh.V_.cols()) };

		for (auto iter = t_v.begin(); iter != t_v.end(); ++iter) {
			Eigen::VectorXd dis; Eigen::VectorXi pre;
			dijkstra(mesh.V_.transpose(),mesh.F_.transpose(), vv, *iter, target, dis, pre);
			distance.row(*iter) = dis.transpose();
			previous.row(*iter) = pre.transpose();
		}
		Eigen::MatrixXd graph;
		std::vector<int> temp(t_v.begin(), t_v.end());
		Eigen::Map<Eigen::VectorXi> idx(temp.data(), temp.size());
		igl::slice(distance, idx, idx, graph);
		std::set<std::pair<int, int>> mst;
		kruskal_minimal_spanning_tree(graph, mst);
		
		for (auto it : mst) {
			int src = *(it.first + idx.begin());
			int pre = *(it.second + idx.begin());
			while (pre != src) {
				edge_set.insert({ pre,previous(src, pre) });
				pre = previous(src, pre);
			}	
		}
		return 1;
	}

	int generate_seam(
		const common::tri_mesh& mesh,
		std::set<std::pair<int, int>>& edge_set,
		const int r, const double percentage)
	{
		std::vector<int> terminal_v;
		select_terminal_vertices(mesh, r, percentage, terminal_v);
		gen_minimal_steiner_tree(mesh, terminal_v, edge_set);
		return 1;
	}
}