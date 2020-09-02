#include "kinectdatatypes/RGBDOperations.h"

#include <cassert>
#include <cmath>

namespace kinectdatatypes {

bool RGBDOperations::getXYZ(const int r, const int c, const float depth,
							const float cx, const float cy,
							const float fracfx, const float fracfy,
							float& x, float& y, float& z) {

	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(!std::isfinite(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return false;
	}

	x = (c + 0.5f - cx) * fracfx * depth_val;
	y = depth_val;
	z = -(r + 0.5f - cy) * fracfy * depth_val;

	assert(std::isfinite(x) &&
		   std::isfinite(y) &&
		   std::isfinite(z));
	return true;
}

bool RGBDOperations::getXYZ(const float r, const float c, const float depth,
							const float cx, const float cy,
							const float fracfx, const float fracfy,
							float& x, float& y, float& z) {
	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(!std::isfinite(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return false;
	}

	x = (c + 0.5f - cx) * fracfx * depth_val;
	y = depth_val;
	z = -(r + 0.5f - cy) * fracfy * depth_val;

	assert(std::isfinite(x) &&
		   std::isfinite(y) &&
		   std::isfinite(z));
	return true;

}

bool RGBDOperations::getRowCol(const float x, const float y, const float z,
							   const float cx, const float cy,
							   const float fracfx, const float fracfy,
							   float& r, float& c) {
	if(std::isnan(x) || std::isnan(y) || std::isnan(z)) {
		return false;
	}

	r = -(z / fracfy / y) + cy - 0.5;
	c = (x / fracfx / y) + cx - 0.5;
	return true;
}

} // namespace kinectdatatypes
