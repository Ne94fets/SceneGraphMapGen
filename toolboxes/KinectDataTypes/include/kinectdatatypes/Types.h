 
#ifndef KINECTDATATYPES_DETECTION_H
#define KINECTDATATYPES_DETECTION_H

#include "KinectDataTypesExports.h"

#include <image/Img.h>

#include <libfreenect2/libfreenect2.hpp>

using namespace mira;

namespace kinectdatatypes {

template<typename _ImgType>
class MIRA_KINECTDATATYPES_EXPORT NumberedFrame
		: public _ImgType {
public:
	typedef _ImgType ImgType;

public:
	NumberedFrame() {}
	NumberedFrame(int width, int height)
		: ImgType(width, height) {}

	size_t frameNumber() const { return m_frameNumber; }
	size_t& frameNumber() { return m_frameNumber; }

public:
	MIRA_NO_GENERIC_REFLECT_MEMBER(NumberedFrame)

	template<typename BinaryStream>
	void reflect(BinarySerializer<BinaryStream>& r) {
		ImgType::reflect(r);
		r.write(reinterpret_cast<const uint8_t*>(&this->m_frameNumber), sizeof(m_frameNumber));
	}

	template<typename BinaryStream>
	void reflect(BinaryDeserializer<BinaryStream>& r) {
		ImgType::reflect(r);
		r.read(reinterpret_cast<uint8_t*>(&this->m_frameNumber), sizeof(m_frameNumber));
	}

private:
	size_t	m_frameNumber;
};

typedef NumberedFrame<Img<uint8_t, 3>>	RGBImgType;
typedef NumberedFrame<Img<float, 1>>	DepthImgType;

template<typename _Type>
class MIRA_KINECTDATATYPES_EXPORT NumberedType
		: public _Type {
public:
	typedef _Type Type;

public:
	NumberedType() {}

	size_t frameNumber() const { return m_frameNumber; }
	size_t& frameNumber() { return m_frameNumber; }

public:
	MIRA_NO_GENERIC_REFLECT_MEMBER(NumberedType)

	template<typename BinaryStream>
	void reflect(BinarySerializer<BinaryStream>& r) {
		r.write(reinterpret_cast<const uint8_t*>(this), sizeof(NumberedType));
	}

	template<typename BinaryStream>
	void reflect(BinaryDeserializer<BinaryStream>& r) {
		r.read(reinterpret_cast<uint8_t*>(this), sizeof(NumberedType));
	}

private:
	size_t	m_frameNumber;
};


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
