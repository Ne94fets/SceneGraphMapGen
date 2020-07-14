#include "kinectdatatypes/Types.h"

namespace kinectdatatypes {

RegistrationData::RegistrationData() {

}

RegistrationData::RegistrationData(const libfreenect2::Freenect2Device::IrCameraParams& depth_p,
								   const libfreenect2::Freenect2Device::ColorCameraParams& rgb_p)
	: depth_p(depth_p),
	  rgb_p(rgb_p) {

}

} // namespace kinectdatatypes
