#ifndef RECOGNITIONDATATYPES_DETECTION_H
#define RECOGNITIONDATATYPES_DETECTION_H

#include "RecognitionDataTypesExports.h"

#include <map>

#include <serialization/BinarySerializer.h>
#include <serialization/NoGenericReflect.h>

#include <opencv2/core.hpp>

using namespace mira;

namespace recognitiondatatypes {

class MIRA_RECOGNITIONDATATYPES_EXPORT Detection {
public:
	Detection();
	Detection(size_t frameNumber,
			  const cv::Rect& box,
			  int type,
			  float confidence,
			  const cv::Point3f pos);

	static std::string getName(int type);

public:
	MIRA_NO_GENERIC_REFLECT_MEMBER(Detection)

	template<typename BinaryStream>
	void reflect(BinarySerializer<BinaryStream>& r) {
		r.write(reinterpret_cast<const uint8_t*>(this), sizeof(Detection));
	}

	template<typename BinaryStream>
	void reflect(BinaryDeserializer<BinaryStream>& r) {
		r.read(reinterpret_cast<uint8_t*>(this), sizeof(Detection));
	}

public:
	size_t		frameNumber;
	cv::Rect	box;
	int			type;
	float		confidence;
	cv::Point3f	pos;

private:
	static std::map<int, std::string> m_lookupMap;
};

} // namespace recognitiondatatypes

#endif // RECOGNITIONDATATYPES_DETECTION_H
