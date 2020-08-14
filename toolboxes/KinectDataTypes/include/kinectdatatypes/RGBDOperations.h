#ifndef RGBDOPERATIONS_H
#define RGBDOPERATIONS_H

#include "KinectDataTypesExports.h"

namespace kinectdatatypes {

class MIRA_KINECTDATATYPES_EXPORT RGBDOperations {
public:
	template<typename PointType>
	static bool getXYZ(const int r, const int c, const float depth,
					   const float cx, const float cy,
					   const float fracfx, const float fracfy,
					   PointType& point) {
		return getXYZ(r, c, depth, cx, cy, fracfx, fracfy, point.x, point.y, point.z);
	}

	template<typename PointType>
	static bool getXYZ(const float r, const float c, const float depth,
					   const float cx, const float cy,
					   const float fracfx, const float fracfy,
					   PointType& point) {
		return getXYZ(r, c, depth, cx, cy, fracfx, fracfy, point.x, point.y, point.z);
	}

	static bool getXYZ(const int r, const int c, const float depth,
					   const float cx, const float cy,
					   const float fracfx, const float fracfy,
					   float& x, float& y, float& z);

	static bool getXYZ(const float r, const float c, const float depth,
					   const float cx, const float cy,
					   const float fracfx, const float fracfy,
					   float& x, float& y, float& z);

};

} // namespace kinectdatatypes

#endif // RGBDOPERATIONS_H
