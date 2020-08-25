#include "recognitiondatatypes/Detection.h"

namespace recognitiondatatypes {


Detection::Detection() {

}

Detection::Detection(const cv::Rect2f& box,
					 int type,
					 float confidence,
					 const cv::Point3f& pos,
					 const cv::Point3f& bboxMin,
					 const cv::Point3f& bboxMax)
	: box(box),
	  type(type),
	  confidence(confidence),
	  pos(pos),
	  bboxMin(bboxMin),
	  bboxMax(bboxMax) {

}

cv::Rect2f Detection::boxOnImage(const cv::Size2i& imgSize, cv::Rect2f box) {
	return cv::Rect2f(box.x * imgSize.width,
					  box.y * imgSize.height,
					  box.width * imgSize.width,
					  box.height * imgSize.height);
}

cv::Rect2f Detection::normalizeBox(const cv::Size2i& imgSize, cv::Rect2f box) {
	return cv::Rect2f(box.x / imgSize.width,
					  box.y / imgSize.height,
					  box.width / imgSize.width,
					  box.height / imgSize.height);
}

const std::string& Detection::getTypeName(int type) {
	auto iter = m_lookupMap.find(type);
	if(iter == m_lookupMap.end())
		throw std::runtime_error("type not found");

	return iter->second;
}

std::unordered_map<int, std::string> Detection::m_lookupMap = {
	{1,	"person"},
	{2,	"bicycle"},
	{3,	"car"},
	{4,	"motorcycle"},
	{5,	"airplane"},
	{6,	"bus"},
	{7,	"train"},
	{8,	"truck"},
	{9,	"boat"},
	{10,	"traffic light"},
	{11,	"fire hydrant"},
	{12,	"street sign"},
	{13,	"stop sign"},
	{14,	"parking meter"},
	{15,	"bench"},
	{16,	"bird"},
	{17,	"cat"},
	{18,	"dog"},
	{19,	"horse"},
	{20,	"sheep"},
	{21,	"cow"},
	{22,	"elephant"},
	{23,	"bear"},
	{24,	"zebra"},
	{25,	"giraffe"},
	{26,	"hat"},
	{27,	"backpack"},
	{28,	"umbrella"},
	{29,	"shoe"},
	{30,	"eye glasses"},
	{31,	"handbag"},
	{32,	"tie"},
	{33,	"suitcase"},
	{34,	"frisbee"},
	{35,	"skis"},
	{36,	"snowboard"},
	{37,	"sports ball"},
	{38,	"kite"},
	{39,	"baseball bat"},
	{40,	"baseball glove"},
	{41,	"skateboard"},
	{42,	"surfboard"},
	{43,	"tennis racket"},
	{44,	"bottle"},
	{45,	"plate"},
	{46,	"wine glass"},
	{47,	"cup"},
	{48,	"fork"},
	{49,	"knife"},
	{50,	"spoon"},
	{51,	"bowl"},
	{52,	"banana"},
	{53,	"apple"},
	{54,	"sandwich"},
	{55,	"orange"},
	{56,	"broccoli"},
	{57,	"carrot"},
	{58,	"hot dog"},
	{59,	"pizza"},
	{60,	"donut"},
	{61,	"cake"},
	{62,	"chair"},
	{63,	"couch"},
	{64,	"potted plant"},
	{65,	"bed"},
	{66,	"mirror"},
	{67,	"dining table"},
	{68,	"window"},
	{69,	"desk"},
	{70,	"toilet"},
	{71,	"door"},
	{72,	"tv"},
	{73,	"laptop"},
	{74,	"mouse"},
	{75,	"remote"},
	{76,	"keyboard"},
	{77,	"cell phone"},
	{78,	"microwave"},
	{79,	"oven"},
	{80,	"toaster"},
	{81,	"sink"},
	{82,	"refrigerator"},
	{83,	"blender"},
	{84,	"book"},
	{85,	"clock"},
	{86,	"vase"},
	{87,	"scissors"},
	{88,	"teddy bear"},
	{89,	"hair drier"},
	{90,	"toothbrush"},
	{91,	"hair brush"}
};

DetectionContainer::DetectionContainer() {

}

} // namespace recognitiondatatypes
