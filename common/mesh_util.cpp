#include "mesh_util.h"

#define ANSI_DECLARATORS
#define REAL double
#define VOID int
#include <triangle/triangle.h>

namespace common {

    int cal_cot_angles(
        const Eigen::MatrixXd& V,
        const Eigen::Matrix3Xi& F,
        Eigen::Matrix3Xd& cot_angles);

    int cal_cot_laplace(
        const Eigen::Matrix3Xd& V,
        const Eigen::Matrix3Xi& F,
        Eigen::SparseMatrix<double>& L)
    {
        Eigen::Matrix3Xd cot_angles;
        cal_cot_angles(V, F, cot_angles);
        std::vector<Eigen::Triplet<double>> triple;
        triple.reserve(F.cols() * 9);
        for (size_t j = 0; j < F.cols(); ++j) {
            const Eigen::Vector3i& fv = F.col(j);
            const Eigen::Vector3d& ca = cot_angles.col(j);
            for (size_t vi = 0; vi < 3; ++vi) {
                const size_t j1 = (vi + 1) % 3;
                const size_t j2 = (vi + 2) % 3;
                const int fv0 = fv[vi];
                const int fv1 = fv[j1];
                const int fv2 = fv[j2];
                triple.push_back(Eigen::Triplet<double>(fv0, fv0, ca[j1] + ca[j2]));
                triple.push_back(Eigen::Triplet<double>(fv0, fv1, -ca[j2]));
                triple.push_back(Eigen::Triplet<double>(fv0, fv2, -ca[j1]));
            }
        }
        L.resize(V.cols(), V.cols());
        L.setFromTriplets(triple.begin(), triple.end());
        return 1;
    }

    int cal_cot_angles(
        const Eigen::MatrixXd& V,
        const Eigen::Matrix3Xi& F,
        Eigen::Matrix3Xd& cot_angles) 
    {
        cot_angles.resize(3, F.cols());
        for (size_t j = 0; j < F.cols(); ++j) {
            const Eigen::Vector3i& fv = F.col(j);
            for (size_t vi = 0; vi < 3; ++vi) {
                const Eigen::VectorXd& p0 = V.col(fv[vi]);
                const Eigen::VectorXd& p1 = V.col(fv[(vi + 1) % 3]);
                const Eigen::VectorXd& p2 = V.col(fv[(vi + 2) % 3]);
                const double angle = std::acos(std::max(-1.0,
                    std::min(1.0, (p1 - p0).normalized().dot((p2 - p0).normalized()))));
                cot_angles(vi, j) = 1.0 / std::tan(angle);
            }
        }
        return 1;
    }

    int convert_uv_to_3d(
        const Eigen::Matrix2Xd& uv,
        Eigen::Matrix3Xd& V)
    {
        V.setZero(3, uv.cols());
        V.topRows(2) = uv;
        return 1;
    }

    int triangulate_polygon(
        const Eigen::Matrix2Xd& V,
        const Eigen::Matrix2Xi& E,
        const Eigen::Matrix2Xd& H,
        double max_area,
        double min_angle,
        Eigen::Matrix2Xd& NV,
        Eigen::Matrix3Xi& T)
    {
        std::stringstream para;
        para << "a" << max_area << "q" << min_angle << "YpzB";//s
        const std::string& full_flags = para.str();

        // Prepare the input struct
        triangulateio in;
        in.numberofpoints = V.cols();
        in.pointlist = const_cast<double*>(V.data());

        in.numberofpointattributes = 0;
        in.pointmarkerlist = (int*)calloc(V.cols(), sizeof(int));
        std::fill(in.pointmarkerlist, in.pointmarkerlist + V.cols(), 1);

        in.trianglelist = NULL;
        in.triangleattributelist = NULL;
        in.numberoftriangles = in.numberofcorners = in.numberoftriangleattributes = 0;

        in.numberofsegments = E.cols();
        in.segmentlist = const_cast<int*>(E.data());
        in.segmentmarkerlist = (int*)calloc(E.cols(), sizeof(int));
        std::fill(in.segmentmarkerlist, in.segmentmarkerlist + E.cols(), 1);

        in.numberofholes = H.cols();
        in.holelist = const_cast<double*>(H.data());
        in.numberofregions = 0;

        // Prepare the output struct
        triangulateio out;
        out.pointlist = NULL;
        out.trianglelist = out.segmentlist = NULL;
        // Call triangulate
        ::triangulate(const_cast<char*>(full_flags.c_str()), &in, &out, 0);

        // Return the mesh
        NV.resize(2, out.numberofpoints);
        std::copy(out.pointlist, out.pointlist + NV.size(), NV.data());
        T.resize(3, out.numberoftriangles);
        std::copy(out.trianglelist, out.trianglelist + T.size(), T.data());

        // Cleanup in & out
        free(in.pointmarkerlist);
        free(in.segmentmarkerlist);
        free(out.pointlist);
        free(out.trianglelist);
        free(out.segmentlist);
        return 0;
    }
}