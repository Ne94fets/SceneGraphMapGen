 
#ifndef KINECTDATATYPES_DETECTION_H
#define KINECTDATATYPES_DETECTION_H

#include "KinectDataTypesExports.h"

#include <image/Img.h>

#include <libfreenect2/libfreenect2.hpp>

using namespace mira;

namespace kinectdatatypes {

class MIRA_KINECTDATATYPES_EXPORT RegistrationData {
public:
	RegistrationData();
	RegistrationData(const libfreenect2::Freenect2Device::IrCameraParams& depth_p,
					 const libfreenect2::Freenect2Device::ColorCameraParams& rgb_p);

public:
	MIRA_NO_GENERIC_REFLECT_MEMBER(RegistrationData)

	template<typename BinaryStream>
	void reflect(BinarySerializer<BinaryStream>& r) {
		r.write(reinterpret_cast<const uint8_t*>(this), sizeof(RegistrationData));
	}

	template<typename BinaryStream>
	void reflect(BinaryDeserializer<BinaryStream>& r) {
		r.read(reinterpret_cast<uint8_t*>(this), sizeof(RegistrationData));
	}

public:
	libfreenect2::Freenect2Device::IrCameraParams		depth_p;
	libfreenect2::Freenect2Device::ColorCameraParams	rgb_p;
};

} // namespace kinectdatatypes

#endif // KINECTDATATYPES_DETECTION_H
