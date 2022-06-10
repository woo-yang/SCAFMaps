#pragma once

#include <Eigen/Dense>

namespace io
{
    int  read_obj(const char* file_name,
        Eigen::Matrix3Xd& V,
        Eigen::Matrix3Xi& F);
    int  save_obj(const char* file_name,
        const Eigen::Matrix3Xd& V,
        const Eigen::Matrix3Xi& F);

    int save_obj(const char* file_name,
        const Eigen::Matrix3Xd& V,
        const Eigen::Matrix3Xi& F,
        const Eigen::Matrix2Xd& UV);
}
