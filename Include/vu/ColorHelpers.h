#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Core>

#include <vu/EigenHelpers.h>

namespace vu {

std::vector<Vec3 < unsigned char> > GetColorPalette(const int K);

} // namespace vu