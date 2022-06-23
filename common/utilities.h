#pragma once
#include <vector>

namespace common {
	class UnionFind {
	public:
		UnionFind(const int n) {
			for (int i = 0; i < n; ++i){
				parent.push_back(i);
				rank.push_back(1);
			}
		}
		int find(const int x)
		{
			if (x == parent[x]) 
				return x;
			else 
				return find(parent[x]);
		}
		inline void merge(const int x, const int y)
		{
			int x_root = find(x), y_root = find(y);
			if (rank[x_root] > rank[y_root]) {
				parent[y_root] = x_root;
			}
			else if (rank[y_root] > rank[x_root]) {
				parent[x_root] = y_root;
			}
			else {
				parent[y_root] = x_root;
				rank[x_root] += 1;
			}
		}
	private:
		std::vector<int> parent;
		std::vector<int> rank;
	};

}

