/*
 * Copyright (C) by
 *   MetraLabs GmbH (MLAB), GERMANY
 * and
 *   Neuroinformatics and Cognitive Robotics Labs (NICR) at TU Ilmenau, GERMANY
 * All rights reserved.
 *
 * Contact: info@mira-project.org
 *
 * Commercial Usage:
 *   Licensees holding valid commercial licenses may use this file in
 *   accordance with the commercial license agreement provided with the
 *   software or, alternatively, in accordance with the terms contained in
 *   a written agreement between you and MLAB or NICR.
 *
 * GNU General Public License Usage:
 *   Alternatively, this file may be used under the terms of the GNU
 *   General Public License version 3.0 as published by the Free Software
 *   Foundation and appearing in the file LICENSE.GPL3 included in the
 *   packaging of this file. Please review the following information to
 *   ensure the GNU General Public License version 3.0 requirements will be
 *   met: http://www.gnu.org/copyleft/gpl.html.
 *   Alternatively you may (at your option) use any later version of the GNU
 *   General Public License if such license has been publicly approved by
 *   MLAB and NICR (or its successors, if any).
 *
 * IN NO EVENT SHALL "MLAB" OR "NICR" BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
 * THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF "MLAB" OR
 * "NICR" HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * "MLAB" AND "NICR" SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND "MLAB" AND "NICR" HAVE NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR MODIFICATIONS.
 */

/**
 * @file SUNRGBDEval.C
 *    Reads SUNRGBD data set and evaluates
 *
 * @author Steffen Kastner
 * @date   2020/08/22
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/Unit.h>
#include <image/Img.h>

#include <kinectdatatypes/Types.h>
#include <recognitiondatatypes/Detection.h>

#include <sstream>

// robot has old gcc need to use experimental here
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <rapidjson/document.h>
#include <rapidjson/reader.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

using namespace mira;

namespace evaluation {

///////////////////////////////////////////////////////////////////////////////

/**
 * Reads SUNRGBD data set and evaluates
 */
class SUNRGBDEval : public Unit
{
MIRA_OBJECT(SUNRGBDEval)
public:
	typedef kinectdatatypes::RegistrationData	RegistrationData;

	typedef Img<uint8_t, 3>	RGBImgType;
	typedef Img<float, 1>	DepthImgType;

	typedef recognitiondatatypes::Detection				Detection;
	typedef recognitiondatatypes::DetectionContainer	DetectionContainer;

public:

	SUNRGBDEval();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, Unit);

		r.property("Path to data set", m_path2DataSet, "Path to data set to evaluate", "external/evalData/SUNRGBD/kv2/align_kv2/");
		r.property("Start", m_start, "Start evaluation", false);
	}

protected:

	virtual void initialize();

	virtual void process(const Timer& timer);

private:

	void onObjectDetection(ChannelRead<DetectionContainer> detections);
	void writeOutputs();
	void addCategories();
	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

	static int SUNRGBDname2typeID(const std::string& name);

private:
	bool			m_start = false;
	bool			m_initRound = false;
	bool			m_rerun = false;
	std::string		m_path2DataSet;

	float	m_IoU2D = 0.5;
	float	m_IoU3D = 0.25;

	std::unordered_map<int, std::string>		m_revName2IDMap;
	std::unordered_map<int, std::vector<float>>	m_rmsePerCategory;

	size_t			m_gtAnnID = 0;
	size_t			m_imgSequenceID = 0;
	size_t			m_loadedSequenceID = 1;
	std::ofstream	m_gtOut;
	std::ofstream	m_detOut;
	rapidjson::Document m_gtDoc;
	rapidjson::Document m_detDoc;

	RegistrationData	m_regData;

	RGBImgType				m_rgb;
	DepthImgType			m_depth;

	fs::directory_iterator	m_dirIter;

	Channel<RegistrationData>	m_channelRegistrationData;
	Channel<RGBImgType>			m_channelRGBFull;	// uchar range: [0,255]
	Channel<DepthImgType>		m_channelDepthFull;	// float unit: [mm] Non-positive, NaN, and infinity are invalid or missing data.
	//Channel<Img<>> mChannel;

	static std::unordered_map<std::string, int> m_SUNRGBDName2COCOIDMap;
};

///////////////////////////////////////////////////////////////////////////////

SUNRGBDEval::SUNRGBDEval() : Unit(Duration::milliseconds(3000)) {
	m_regData.rgb_p.cx = 365.0;
	m_regData.rgb_p.cy = 265.0;
	m_regData.rgb_p.fx = 529.5;
	m_regData.rgb_p.fy = 529.5;
	// TODO: further initialization of members, etc.

	for(const auto& pair : m_SUNRGBDName2COCOIDMap) {
		m_revName2IDMap[pair.second] = pair.first;
		m_rmsePerCategory[pair.second] = std::vector<float>();
	}
}

void SUNRGBDEval::initialize() {
	m_channelRegistrationData = publish<RegistrationData>("KinectRegData");
	m_channelRGBFull = publish<RGBImgType>("RGBImageFull");
	m_channelDepthFull = publish<DepthImgType>("DepthImageFull");

	subscribe<DetectionContainer>("ObjectDetectionNet", &SUNRGBDEval::onObjectDetection);
}

void SUNRGBDEval::process(const Timer& timer) {
	if(!m_start || (m_imgSequenceID == m_loadedSequenceID && m_rerun)) {
		return;
	}

	if(!m_initRound) {
		m_initRound = true;

		m_imgSequenceID = 0;
		m_gtAnnID = 0;

		// open output file
		std::string path = m_path2DataSet + "gtOut.json";
		m_gtOut.open(path);
		if(!m_gtOut.is_open()) {
			throw std::runtime_error("Could not open: " + path);
		}

		m_gtDoc.SetObject();
		m_gtDoc.AddMember("annotations", rapidjson::Value(rapidjson::kArrayType), m_gtDoc.GetAllocator());
		m_gtDoc.AddMember("images", rapidjson::Value(rapidjson::kArrayType), m_gtDoc.GetAllocator());
		addCategories();

		path = m_path2DataSet + "detOut.json";
		m_detOut.open(path);
		if(!m_detOut.is_open()) {
			throw std::runtime_error("Could not open: " + path);
		}

		m_detDoc.SetArray();

		m_dirIter = fs::directory_iterator(m_path2DataSet);
	}

	while(m_dirIter != fs::end(m_dirIter) && !fs::is_directory(m_dirIter->path())) {
		++m_dirIter;
	}

	if(m_dirIter == fs::end(m_dirIter)) {
		m_start = false;
		m_initRound = false;
		writeOutputs();
		m_gtOut.close();
		m_detOut.close();
		return;
	}

	const auto captureTime = Time::now();

	auto wRegData = m_channelRegistrationData.write();
	wRegData->sequenceID = m_imgSequenceID;
	wRegData->timestamp = captureTime;
	wRegData->value() = m_regData;

	cv::Mat tmpRGB, tmpDepth;
	std::string imageDir = std::string(m_dirIter->path()) + "/image/";
	for(const auto& entry : fs::directory_iterator(imageDir)) {
		std::string filePath = entry.path();
		if(0 == filePath.compare(filePath.length() - 3, 3, "jpg")) {
			tmpRGB = cv::imread(filePath);
		} else if(0 == filePath.compare(filePath.length() - 3, 3, "png")) {
			tmpDepth = cv::imread(filePath, cv::IMREAD_ANYDEPTH);
		}
	}
	std::string depthDir = std::string(m_dirIter->path()) + "/depth/";
	for(const auto& entry : fs::directory_iterator(depthDir)) {
		std::string filePath = entry.path();
		if(0 == filePath.compare(filePath.length() - 3, 3, "jpg")) {
			tmpRGB = cv::imread(filePath);
		} else if(0 == filePath.compare(filePath.length() - 3, 3, "png")) {
			tmpDepth = cv::imread(filePath, cv::IMREAD_ANYDEPTH);
		}
	}

	if(tmpRGB.empty() || tmpDepth.empty()) {
		m_start = false;
		m_initRound = false;
		writeOutputs();
		m_gtOut.close();
		m_detOut.close();
		return;
	}

	tmpRGB.copyTo(m_rgb);
//	cv::resize(tmpRGB, rgb, cv::Size2i(1920, 1080), 0, 0, cv::INTER_CUBIC);
	// convert 16-bit shifted img to float, took from read3dPoints.m in SUNRGBDtoolbox
	for(auto pixIter = tmpDepth.begin<ushort>(); pixIter != tmpDepth.end<ushort>(); ++pixIter) {
		ushort pix = *pixIter;
		*pixIter = (pix >> 3) | (pix << 13);
	}
//	cv::Mat tmpDepthBig;
//	cv::resize(tmpDepth, tmpDepthBig, cv::Size2i(1920, 1080), 0, 0, cv::INTER_NEAREST);
	tmpDepth.convertTo(m_depth, CV_32F);

	ChannelWrite<RGBImgType> wRGBFull = m_channelRGBFull.write();
	wRGBFull->sequenceID = m_imgSequenceID;
	wRGBFull->timestamp = captureTime;
	wRGBFull->value() = m_rgb;

	ChannelWrite<DepthImgType> wDepthFull = m_channelDepthFull.write();
	wDepthFull->sequenceID = m_imgSequenceID;
	wDepthFull->timestamp = captureTime;
	wDepthFull->value() = m_depth;

	if(m_imgSequenceID == m_loadedSequenceID) {
		m_rerun = true;
	} else {
		m_rerun = false;
	}
}

void SUNRGBDEval::onObjectDetection(ChannelRead<SUNRGBDEval::DetectionContainer> detections) {
	// resend image to clear background detections
	const auto captureTime = Time::now();
	ChannelWrite<RGBImgType> wRGBFull = m_channelRGBFull.write();
	wRGBFull->sequenceID = m_imgSequenceID;
	wRGBFull->timestamp = captureTime;
	wRGBFull->value() = m_rgb;

	ChannelWrite<DepthImgType> wDepthFull = m_channelDepthFull.write();
	wDepthFull->sequenceID = m_imgSequenceID;
	wDepthFull->timestamp = captureTime;
	wDepthFull->value() = m_depth;

	cv::Mat currentImg = m_rgb;
	cv::Size2i imgSize = cv::Size(currentImg.cols, currentImg.rows);
	size_t currentSequenceID = detections->sequenceID;

	std::string currentDir = m_dirIter->path();
	m_dirIter++;
	m_imgSequenceID++;

	std::string filePath = currentDir +
			"/annotation2D3D/index.json";

	std::fstream annoFile(filePath);
	if(!annoFile.is_open()) {
		throw std::runtime_error("Could not open annotations file: " + filePath);
	}

	std::string content;
	annoFile.seekg(0, std::ios::end);
	content.reserve(annoFile.tellg());
	annoFile.seekg(0, std::ios::beg);

	content.assign((std::istreambuf_iterator<char>(annoFile)), std::istreambuf_iterator<char>());

	rapidjson::Document d;
	d.Parse(content.c_str(), content.length());

	std::unordered_map<int, std::string> names;
	std::unordered_map<int, Detection> gtDetections;

	// read ground truth detections
	if(!d.HasMember("objects") || !d.HasMember("frames")) {
		std::cout << "no member objects/frames in " << detections->sequenceID << std::endl;
		return;
	}

	rapidjson::Value& frames = *d["frames"].Begin();
	if(!frames.HasMember("polygon")) { return; }

	rapidjson::Value& objects = d["objects"];
	size_t objId = 0;
	for(auto objIter = objects.Begin(); objIter != objects.End(); ++objIter) {
		rapidjson::Value& obj = *objIter;
		Detection det;
		if(obj.IsNull() ||
				!obj.HasMember("name") ||
				!obj.HasMember("polygon")){
			objId++;
			continue;
		}

		std::string name = obj["name"].GetString();
		Eigen::Vector4d pos(0,0,0,1);

		rapidjson::Value& polygon = obj["polygon"];
		for(auto polyIter = polygon.Begin(); polyIter != polygon.End(); ++polyIter) {
			rapidjson::Value& polyElem = *polyIter;
			det.pos.y = (polyElem["Ymin"].GetFloat() + polyElem["Ymax"].GetFloat())/2;

			size_t xCnt = 0;
			for(auto xIter = polyElem["X"].Begin(); xIter != polyElem["X"].End(); ++xIter) {
				pos.x() += xIter->GetFloat();
				xCnt++;
			}
			pos.x() /= xCnt;

			size_t zCnt = 0;
			for(auto zIter = polyElem["Z"].Begin(); zIter != polyElem["Z"].End(); ++zIter) {
				pos.z() += zIter->GetFloat();
				zCnt++;
			}
			pos.z() /= zCnt;
		}

		// y-down, x-right to z-up, x-right
		det.pos = cv::Point3f(pos.x(), pos.z(), -pos.y());

		names[objId] = name;
		gtDetections[objId++] = det;
	}

	rapidjson::Value& polygon = frames["polygon"];
	for(auto polyIter = polygon.Begin(); polyIter != polygon.End(); ++polyIter) {
		rapidjson::Value& poly = *polyIter;
		if(poly.IsNull() ||
				!poly.HasMember("object") ||
				!poly.HasMember("x") ||
				!poly.HasMember("y")) {
			std::cout << "Parsing 2D polygon failed" << std::endl;
			continue;
		}

		int objId = poly["object"].GetInt();
		if(gtDetections.find(objId) == gtDetections.end()) {
			continue;
		}

		rapidjson::Value& xCoords = poly["x"];
		rapidjson::Value& yCoords = poly["y"];
		float xmin, ymin, xmax, ymax;
		xmin = ymin = std::numeric_limits<float>::max();
		xmax = ymax = std::numeric_limits<float>::min();
		for(auto xIter = xCoords.Begin(); xIter != xCoords.End(); ++xIter) {
			float xVal = xIter->GetFloat();
			xmin = std::min(xmin, xVal);
			xmax = std::max(xmax, xVal);
		}
		for(auto yIter = yCoords.Begin(); yIter != yCoords.End(); ++yIter) {
			float yVal = yIter->GetFloat();
			ymin = std::min(ymin, yVal);
			ymax = std::max(ymax, yVal);
		}
		gtDetections[objId].box = cv::Rect2f(xmin, ymin, xmax-xmin, ymax-ymin);

		cv::rectangle(currentImg, gtDetections[objId].box, cv::Scalar(0, 0, 255), 4);
	}


	// write gt to gt document
	rapidjson::Document::AllocatorType& gtAlloc = m_gtDoc.GetAllocator();
	rapidjson::Value annImg(rapidjson::kObjectType);
	annImg.AddMember("id", rapidjson::Value().SetInt(currentSequenceID), gtAlloc);
	m_gtDoc["images"].PushBack(annImg, gtAlloc);
	for(const auto& pair : gtDetections) {
		rapidjson::Value jsonGTObj(rapidjson::kObjectType);
		try {
			int gtCategoryID = SUNRGBDname2typeID(names[pair.first]);
			jsonGTObj.AddMember("id", rapidjson::Value().SetInt(m_gtAnnID++), gtAlloc);
			jsonGTObj.AddMember("image_id", rapidjson::Value().SetInt(currentSequenceID), gtAlloc);
			jsonGTObj.AddMember("category_id", rapidjson::Value().SetInt(gtCategoryID), gtAlloc);
			jsonGTObj.AddMember("iscrowd", rapidjson::Value().SetInt(0), gtAlloc);
			jsonGTObj.AddMember("area", rapidjson::Value().SetFloat(pair.second.box.area()), gtAlloc);

			rapidjson::Value bbox(rapidjson::kArrayType);
			bbox.PushBack(rapidjson::Value(pair.second.box.x), gtAlloc);
			bbox.PushBack(rapidjson::Value(pair.second.box.y), gtAlloc);
			bbox.PushBack(rapidjson::Value(pair.second.box.width), gtAlloc);
			bbox.PushBack(rapidjson::Value(pair.second.box.height), gtAlloc);
			jsonGTObj.AddMember("bbox", bbox, gtAlloc);

			m_gtDoc["annotations"].PushBack(jsonGTObj, gtAlloc);
		} catch (std::runtime_error e) {
			std::cout << e.what() << std::endl;
		}
	}

	// write det to det document
	rapidjson::Document::AllocatorType& detAlloc = m_detDoc.GetAllocator();
	for(const auto& det : *detections) {
		auto resizedBox = Detection::boxOnImage(imgSize, det.box);

		rapidjson::Value jsonDetObj(rapidjson::kObjectType);
		try {
			jsonDetObj.AddMember("image_id", rapidjson::Value().SetInt(currentSequenceID), detAlloc);
			jsonDetObj.AddMember("category_id", rapidjson::Value().SetInt(det.type), detAlloc);

			rapidjson::Value bbox(rapidjson::kArrayType);
			bbox.PushBack(rapidjson::Value(resizedBox.x), gtAlloc);
			bbox.PushBack(rapidjson::Value(resizedBox.y), gtAlloc);
			bbox.PushBack(rapidjson::Value(resizedBox.width), gtAlloc);
			bbox.PushBack(rapidjson::Value(resizedBox.height), gtAlloc);
			jsonDetObj.AddMember("bbox", bbox, detAlloc);

			jsonDetObj.AddMember("score", rapidjson::Value().SetFloat(det.confidence), detAlloc);

			m_detDoc.PushBack(jsonDetObj, detAlloc);
		} catch(std::runtime_error e) {
			std::cout << e.what() << std::endl;
		}

		float maxIoU = 0;
		std::pair<int, Detection> matchGT;
		for(const auto& pair : gtDetections) {
			const std::string& gtName = names[pair.first];
			const auto iter = m_SUNRGBDName2COCOIDMap.find(gtName);
			if(iter == m_SUNRGBDName2COCOIDMap.end() || iter->second != det.type) {
				continue;
			}

			float iou2D = Detection::calcIoU(resizedBox, pair.second.box);
			if(iou2D > maxIoU) {
				maxIoU = iou2D;
				matchGT = pair;
			}
		}

		if(maxIoU <= 0) {
			std::cout << "No match for: " << Detection::getTypeName(det.type) << std::endl;
			continue;
		}

		std::cout << "Comparing " << Detection::getTypeName(det.type) << " with " << names[matchGT.first] << std::endl;

		auto det2GT = det.pos - matchGT.second.pos;
		float dist = det2GT.dot(det2GT);

		m_rmsePerCategory[det.type].push_back(dist);
	}

	cv::imshow("Eval SUNRGBD", currentImg);
}

void SUNRGBDEval::writeOutputs() {
	rapidjson::OStreamWrapper gtFWS(m_gtOut);
	rapidjson::Writer<rapidjson::OStreamWrapper> gtWriter(gtFWS);
	m_gtDoc.Accept(gtWriter);

	rapidjson::OStreamWrapper detFWS(m_detOut);
	rapidjson::Writer<rapidjson::OStreamWrapper> detWriter(detFWS);
	m_detDoc.Accept(detWriter);

	std::ofstream rmseFile(m_path2DataSet + "/rmse200.json");
	rmseFile << "{";
	size_t cnt = 0;
	for(const auto& pair : m_rmsePerCategory) {
		rmseFile << "\"" << Detection::getTypeName(pair.first) << "\":[";
		for(size_t i = 0; i < pair.second.size(); ++i) {
			auto dist = pair.second[i];
			if(!std::isfinite(dist)) {
				continue;
			}
			rmseFile << dist;
			if(i != pair.second.size()-1) {
				rmseFile << ",";
			}
		}
		rmseFile << "]";
		if(cnt++ != m_rmsePerCategory.size()-1) {
			rmseFile << ",";
		}
	}
	rmseFile << "}";
	rmseFile.close();
}

void SUNRGBDEval::addCategories() {
	rapidjson::Document::AllocatorType& gtAlloc = m_gtDoc.GetAllocator();
	rapidjson::Value categories(rapidjson::kArrayType);
	for(const auto& pair : m_SUNRGBDName2COCOIDMap) {
		rapidjson::Value cat(rapidjson::kObjectType);
		cat.AddMember("supercategory", rapidjson::Value().SetString("SUNRGBD", 7), gtAlloc);
		cat.AddMember("id", rapidjson::Value().SetInt(pair.second), gtAlloc);
		const std::string& origName = Detection::getTypeName(pair.second);
		cat.AddMember("name", rapidjson::Value().SetString(origName.c_str(), origName.length()), gtAlloc);
		categories.PushBack(cat, gtAlloc);
	}
	m_gtDoc.AddMember("categories", categories, gtAlloc);
}

int SUNRGBDEval::SUNRGBDname2typeID(const std::string& name) {
	auto iter = m_SUNRGBDName2COCOIDMap.find(name);
	if(iter == m_SUNRGBDName2COCOIDMap.end()) {
		throw std::runtime_error("name not found: " + name);
	}

	return iter->second;
}

std::unordered_map<std::string, int> SUNRGBDEval::m_SUNRGBDName2COCOIDMap = {
	{"bed", 65},
	{"bench", 15},
	{"book", 84},
	{"chair", 62},
	{"stool", 62},
	{"sofa_chair", 62}, // 63 => 0.205, // 62 => 0.233
	{"sofa", 63},
	{"desk", 69},
	{"dining_table", 67},
	{"endtable", 67},
	{"table", 67},
	{"door", 71},
	{"keyboard", 76},
	{"microwave", 78},
	{"mouse", 74},
	{"oven", 79},
	{"person", 1},
	{"flower_pot", 64},
	{"flower_vase", 64},
	{"plant", 64},
	{"plants", 64},
	{"sink", 81},
	{"toilet", 70},
	{"urinal", 70},
	{"monitor", 72},
	{"tv", 72},
	{"fridge", 82},
};

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(evaluation::SUNRGBDEval, mira::Unit);
