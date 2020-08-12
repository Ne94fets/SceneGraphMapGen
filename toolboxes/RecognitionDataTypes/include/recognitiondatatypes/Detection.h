#ifndef RECOGNITIONDATATYPES_DETECTION_H
#define RECOGNITIONDATATYPES_DETECTION_H

#include "RecognitionDataTypesExports.h"

#include <unordered_map>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <serialization/BinarySerializer.h>
#include <serialization/NoGenericReflect.h>

#include <opencv2/core.hpp>

using namespace mira;

namespace recognitiondatatypes {

class MIRA_RECOGNITIONDATATYPES_EXPORT Detection {
public:
	Detection();
	Detection(const cv::Rect2f& box,
			  int type,
			  float confidence,
			  const cv::Point3f& pos,
			  const cv::Point3f& bboxMin,
			  const cv::Point3f& bboxMax);

	static std::string getTypeName(int type);

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
	size_t				pyramidLevel;
	cv::Rect2f			box;
	int					type;
	float				confidence;
	cv::Point3f			pos;
	cv::Point3f			bboxMin;
	cv::Point3f			bboxMax;
	boost::uuids::uuid	uuid;

private:
	static std::unordered_map<int, std::string> m_lookupMap;
};

class MIRA_RECOGNITIONDATATYPES_EXPORT DetectionContainer
		: public std::vector<Detection> {
public:
	DetectionContainer();

public:
	MIRA_NO_GENERIC_REFLECT_MEMBER(Detection)

	template<typename BinaryStream>
	void reflect(BinarySerializer<BinaryStream>& r) {
		size_t size = this->size();
		r.write(&size, sizeof(size_t));
		for(auto it = this->begin(); it != this->end(); ++it)
			it->reflect(r);
	}

	template<typename BinaryStream>
	void reflect(BinaryDeserializer<BinaryStream>& r) {
		size_t size;
		r.read(&size, sizeof(size_t));
		this->reserve(size);
		for(size_t i = 0; i < size; ++i) {
			Detection d;
			d.reflect(r);
			this->push_back(d);
		}
	}

};

} // namespace recognitiondatatypes

#endif // RECOGNITIONDATATYPES_DETECTION_H
