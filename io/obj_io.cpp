#include "obj_io.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace io {
    int read_obj(const char* file_name,
        Eigen::Matrix3Xd& V,
        Eigen::Matrix3Xi& F)
    {
        std::ifstream is(file_name);
        if (!is)
            return 0;
        std::vector<double>  vs;
        std::vector<int>  fs;
        std::string line, pair[3];
        double  node[3];
        int  tri;
        while (!is.eof()) {
            std::getline(is, line);
            if (line.empty() || 13 == line[0])
                continue;
            std::istringstream instream(line);
            std::string word;
            instream >> word;
            if ("v" == word || "V" == word) {
                instream >> node[0] >> node[1] >> node[2];
                for (size_t j = 0; j < 3; ++j) {
                    vs.push_back(node[j]);
                }
            }
            else if ('f' == word[0] || 'F' == word[0]) {
                instream >> pair[0] >> pair[1] >> pair[2];
                for (size_t j = 0; j < 3; ++j) {
                    tri = strtoul(pair[j].c_str(), NULL, 10) - 1;
                    fs.push_back(tri);
                }
            }
        }
        is.close();
        V = Eigen::Map<Eigen::Matrix3Xd>(vs.data(), 3, vs.size() / 3);
        F = Eigen::Map<Eigen::Matrix3Xi>(fs.data(), 3, fs.size() / 3);
        return 1;
    }

    int save_obj(const char* file_name,
        const Eigen::Matrix3Xd& V,
        const Eigen::Matrix3Xi& F)
    {
        std::ofstream os(file_name);
        if (!os)
            return 0;

        for (int i = 0; i < V.cols(); ++i) {
            os << "v " << V(0, i) << " " << V(1, i) << " " << V(2, i) << "\n";
        }
        for (int i = 0; i < F.cols(); ++i) {
            os << "f " << F(0, i) + 1 << " " << F(1, i) + 1 << " " << F(2, i) + 1 << "\n";
        }
        os.close();
        return 1;
    }

    int save_obj(const char* file_name,
        const Eigen::Matrix3Xd& V,
        const Eigen::Matrix3Xi& F,
        const Eigen::Matrix2Xd& UV)
    {
        std::ofstream os(file_name);
        if (!os)
            return 0;

        Eigen::Matrix2Xd uv = UV;
        Eigen::Vector2d uv_min = uv.rowwise().minCoeff();
        uv.colwise() -= uv_min;
        double uv_len = uv.rowwise().maxCoeff().maxCoeff();
        uv /= uv_len;

        for (int i = 0; i < V.cols(); ++i) {
            os << "v " << V(0, i) << " " << V(1, i) << " " << V(2, i) << "\n";
        }
        for (int i = 0; i < uv.cols(); ++i) {
            os << "vt " << uv(0, i) << " " << uv(1, i) << "\n";
        }
        for (int i = 0; i < F.cols(); ++i) {
            os << "f " << F(0, i) + 1 <<"/" << F(0, i) + 1 << " " 
                       << F(1, i) + 1 <<"/" << F(1, i) + 1 << " " 
                       << F(2, i) + 1 <<"/" << F(2, i) + 1 << "\n";
        }
        os.close();
        return 1;
    }

}
