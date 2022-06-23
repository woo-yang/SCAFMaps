#include "GWindow.h"

int main() {

	YRender::GWindow window("example");
	Eigen::Matrix3d v;
	v << 0., 1., 0., 0, 0, 1, 0, 0, 0;
	Eigen::Vector3i t;
	t << 0, 1, 2;

	window.push_object(v, t, YRender::Y_TRIANGLES);

	window.show();
	return 0;
}