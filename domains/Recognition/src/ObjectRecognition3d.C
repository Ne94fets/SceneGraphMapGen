/*
 * Copyright (C) 2015 by
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
 * @file ObjectRecognition3d.C
 *    Recognizes object in RGB images and locates them in 3D space
 *
 * @author Steffen Kastner
 * @date   2020/02/21
 */

#include "ObjectRecognition3d.h"

#include <assert.h>

#include <algorithm>
#include <exception>
#include <iomanip>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>

#include <utils/Time.h>

#define DEBUG_POS_SAMPLING 0

namespace tf = tensorflow;

using namespace mira;

class Vector2TensorBuffer : public tf::TensorBuffer {
public:
	Vector2TensorBuffer(void* data, std::size_t len) : tf::TensorBuffer(data), m_len(len) {}
	//returns how many bytes we have in our buffer
	std::size_t size() const override {return m_len;}
	//needed so TF knows this isn't a child of some other buffer
	TensorBuffer* root_buffer() override { return this; }
	// Not actually sure why we need this, but it lets TF know where the memory for this tensor came from
	void FillAllocationDescription(tensorflow::AllocationDescription*) const override {}
	// A value of false indicates this TensorBuffer does not own the underlying data
	bool OwnsMemory() const override { return false; }
private:
	std::size_t m_len;
};

namespace recognition {

ObjectRecognition3d::ObjectRecognition3d() {
	// TODO: further initialization of members, etc.{

	// reserve memory for 10000 points used in calcPosition
	m_calcPositionBuffer.reserve(100*100);
}

ObjectRecognition3d::~ObjectRecognition3d() {
	m_shutdown = true;
	delete m_session;

	if(m_trackThread) {
		m_trackThread->join();
		delete m_trackThread;
	}

	if(m_bgThread) {
		m_bgThread->join();
		delete m_bgThread;
	}
}

void ObjectRecognition3d::initialize() {
	tf::Status status = tf::NewSession(tf::SessionOptions(), &m_session);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	tf::GraphDef graphDef;
	status = tf::ReadBinaryProto(tf::Env::Default(), "model.pb", &graphDef);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	// Add the graph to the session
	status = m_session->Create(graphDef);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	subscribe<RegistrationData>("KinectRegData", &ObjectRecognition3d::onRegistrationData);
	subscribe<RGBImgType>("RGBImageFull", &ObjectRecognition3d::onNewRGBImage);
	subscribe<DepthImgType>("DepthImageFull", &ObjectRecognition3d::onNewDepthImage);

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &ObjectRecognition3d::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
	m_channelRGBMarked = publish<RGBImgType>("RGBImageMarked");
	m_channelDetections = publish<DetectionContainer>("ObjectDetection");
	m_channelNewDetections = publish<DetectionContainer>("ObjectDetectionNew");
	m_channelLostDetections = publish<DetectionContainer>("ObjectDetectionLost");

	publishService(*this);

	if(!m_trackThread) {
		m_trackThread = new std::thread([this](){ process(); });
	}

	if(!m_bgThread) {
		m_bgThread = new std::thread([this](){ backgroundProcess(); });
	}
}

void ObjectRecognition3d::onRegistrationData(ChannelRead<ObjectRecognition3d::RegistrationData> data) {
	if(m_hasRegData)
		return;

	m_regData = *data;
	m_hasRegData = true;
}

void ObjectRecognition3d::onNewRGBImage(
		ChannelRead<ObjectRecognition3d::RGBImgType> image) {
	static uint32_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing color image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	m_rgbdQueue.push0(image);
}

void ObjectRecognition3d::onNewDepthImage(
		ChannelRead<ObjectRecognition3d::DepthImgType> image) {
	static size_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing depth image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	m_rgbdQueue.push1(image);
}

void ObjectRecognition3d::process() {
	while(!m_shutdown) {

		auto startTime = std::chrono::system_clock::now();

		// sync queues and get a matching pair
		const auto optionalPair = m_rgbdQueue.getNewestSyncedPair();
		if(!optionalPair) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// got a pair
		const auto& pair = *optionalPair;

		// process the pair
		processPair(pair);

		auto endTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		if(duration > 1000/30)
			std::cout << "Detectionprocess took: " << duration << "ms" << std::endl;
	}
}

void ObjectRecognition3d::backgroundProcess() {
	std::vector<long> durations;
	while(!m_shutdown) {
		if(m_bgStatus == BackgroundStatus::WAITING ||
				m_bgStatus == BackgroundStatus::DONE) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		std::vector<Detection> detections;

		auto startTime = std::chrono::system_clock::now();
		{
			std::lock_guard<std::mutex> imageGuard(m_detectionImageMutex);
			detections = detect(m_detectionImage);
		}
		auto endTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		durations.push_back(duration);

		if(duration > 1000/30) {
			std::cout << "Detecting " << detections.size() << " took: " << duration << "ms" << std::endl;
		}

		m_bgDetections = detections;

		float avg(0);
		for(const auto d : durations) {
			avg += float(d) / durations.size();
		}

		std::cout << "Average detection duration: " << avg << std::endl;

		m_bgStatus = BackgroundStatus::DONE;
	}
}

void ObjectRecognition3d::processPair(const ChannelPair& pair) {
	const Stamped<RGBImgType>& rgbImage = pair.first;
	const Stamped<DepthImgType>& depthImage = pair.second;
	Stamped<RGBImgType> rgbSmall;
	rgbSmall.sequenceID = pair.first.sequenceID;
	cv::resize(rgbImage, rgbSmall, cv::Size(rgbImage.width()/4, rgbImage.height()/4));

	rgbImage.getMat().copyTo(m_currentRGBMarked);

	// clear lost and new detections
	m_detectionsNew.clear();
	m_detectionsLost.clear();

	// start a new detection thread if none is running
	startDetection(pair.first);

	// track last detections
	trackLastDetections(rgbSmall, depthImage);

	// join and clean up old thread if possible and get detections
	trackNewDetections(rgbSmall, depthImage);

	// draw detections to an debug output image
	auto drawStart = std::chrono::system_clock::now();

	for(const auto& d : m_detections) {
		cv::Rect resizedRect = rect2ImageCoords(m_currentRGBMarked, d.box);
		cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), resizedRect, cv::Scalar(0, 0, 255), 4);

		std::stringstream topText;
		try {
			topText << Detection::getTypeName(d.type) << ": " << std::setprecision(0) << std::fixed << d.confidence * 100;
		} catch (const std::runtime_error& e) {
			std::cout << e.what() << std::endl;
		}
		int baseline = 0;
		double fontScale = 1;
		int thickness = 2;
		auto textSize = cv::getTextSize(topText.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
		auto textPos = cv::Point2i(static_cast<int>(d.box.x * m_currentRGBMarked.width()),
								   static_cast<int>(d.box.y * m_currentRGBMarked.height()));
		cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), cv::Scalar(0, 0, 255), -1);
		cv::putText(m_currentRGBMarked.getMat(), topText.str(), textPos,
					cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
		textPos.y += textSize.height + 4;
		std::stringstream textRelPos;
		textRelPos << d.pos;
		cv::putText(m_currentRGBMarked.getMat(), textRelPos.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255), thickness);
	}

	auto drawEnd = std::chrono::system_clock::now();
	auto drawDuration = std::chrono::duration_cast<std::chrono::milliseconds>(drawEnd - drawStart).count();
	if(drawDuration > 1000/30)
		std::cout << "Draw detection took: " << drawDuration << "ms" << std::endl;

//	m_channelRGBMarked.post(Stamped<RGBImgType>(m_currentRGBMarked,
//												Time::now(),
//												pair.first->sequenceID));
	auto wRGBMarked = m_channelRGBMarked.write();
	wRGBMarked->sequenceID = pair.first.sequenceID;
	wRGBMarked->value() = m_currentRGBMarked.clone();

	// post detections
	auto wChannelDetections = m_channelDetections.write();
	wChannelDetections->sequenceID = pair.first.sequenceID;
	wChannelDetections->value() = m_detections;

	auto wChannelNewDetections = m_channelNewDetections.write();
	wChannelNewDetections->sequenceID = pair.first.sequenceID;
	wChannelNewDetections->value() = m_detectionsNew;

	auto wChannelLostDetections = m_channelLostDetections.write();
	wChannelLostDetections->sequenceID = pair.first.sequenceID;
	wChannelLostDetections->value() = m_detectionsLost;
}

void ObjectRecognition3d::startDetection(const Stamped<RGBImgType>& rgbImage) {
	// only start if non is running
	if(m_bgStatus == BackgroundStatus::WAITING) {
		std::lock_guard<std::mutex> imageGuard(m_detectionImageMutex);
		m_detectionImage = rgbImage;

		m_bgStatus = BackgroundStatus::WORKING;
	}
}

void ObjectRecognition3d::trackLastDetections(const Stamped<RGBImgType>& rgbImage,
											  const Stamped<DepthImgType>& depthImage) {
	auto trackingStart = std::chrono::system_clock::now();

	size_t validCnt = 0;
	for(size_t i = 0; i < m_trackers.size(); ++i) {
		cv::Rect2d box;

		auto trackingStart = std::chrono::system_clock::now();
		bool trackSucess = m_trackers[i]->update(rgbImage, box);
		if(trackSucess && box.height > 0 && box.width > 0) {
			auto trackingEnd = std::chrono::system_clock::now();
			auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
			if(trackingDuration > 1000/30)
				std::cout << "Tracking #" << i << " took: " << trackingDuration << "ms" << std::endl;

			// clamp box
			box = clampRect(rgbImage, box);

			// tracked outside of the image if width and height are smaller than zero after clamping
			if(box.width < 1 || box.height < 1) {
				std::cout << "Tracked object #" << i << " is outside of image" << std::endl;
				continue;
			}

			// update detection
			auto normalized = normalizeRect(rgbImage, box);
			auto& d = m_detections[i];
			updateDetectionBox(d, depthImage, normalized);

			// swap detection and tracker to front
			std::swap(m_detections[validCnt], m_detections[i]);
			std::swap(m_trackers[validCnt], m_trackers[i]);
			validCnt++;
		} else {
			auto trackingEnd = std::chrono::system_clock::now();
			auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
			if(trackingDuration > 1000/30)
				std::cout << "Tracking #" << i << " lost took: " << trackingDuration << "ms" << std::endl;
		}
	}

	// move lost trackers to bg trackers and use them again
	m_bgTrackers.insert(m_bgTrackers.end(), m_trackers.begin() + validCnt, m_trackers.end());

	// move lost detections to lostDetections
	m_detectionsLost.insert(m_detectionsLost.end(), m_detections.begin() + validCnt, m_detections.end());

	// remove lost detections and thier trackers from active detections
	m_detections.resize(validCnt);
	m_trackers.resize(validCnt);

	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " detections took: " << trackingDuration << "ms" << std::endl;
}

void ObjectRecognition3d::trackNewDetections(const Stamped<RGBImgType>& rgbImage,
											 const Stamped<DepthImgType>& depthImage) {
	auto trackingStart = std::chrono::system_clock::now();

	// return if the background process is still running
	if(m_bgStatus != BackgroundStatus::DONE) {
		return;
	}

	// update new detections to current image
	if(!m_bgDetections.empty()) {
		// add more trackers if needed
		m_bgTrackers.reserve(m_bgDetections.size());
		for(size_t i = m_bgTrackers.size(); i < m_bgDetections.size(); ++i)
			m_bgTrackers.push_back(cv::TrackerKCF::create());

		cv::Mat resizedDetectionImage;
		cv::resize(m_detectionImage, resizedDetectionImage, rgbImage.size());

		size_t validCnt = 0;
		for(size_t i = 0; i < m_bgDetections.size(); ++i) {
			auto& d = m_bgDetections[i];
			if(d.confidence < m_confidenceThreshold) {
				continue;
			}

			cv::Rect2d box = rect2ImageCoords(resizedDetectionImage, d.box);
			m_bgTrackers[i]->init(resizedDetectionImage, box);

			cv::Rect2d debug = rect2ImageCoords(m_currentRGBMarked, d.box);
			cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), debug, cv::Scalar(255, 0, 255), 4);

			// try to track to current image
			if(!m_bgTrackers[i]->update(rgbImage, box)) {
				continue;
			}

			// clamp box
			box = clampRect(rgbImage, box);

			if(box.width <= 0 || box.height <= 0) {
				continue;
			}

			// update detection
			auto normalized = normalizeRect(rgbImage, box);
			updateDetectionBox(d, depthImage, normalized);

			// swap detection and tracker to front
			std::swap(m_bgDetections[validCnt], m_bgDetections[i]);
			std::swap(m_bgTrackers[validCnt], m_bgTrackers[i]);
			validCnt++;
		}

		// delete lost trackers
		m_bgDetections.resize(validCnt);
		m_bgTrackers.resize(validCnt);

		// move new trackers to foreground trackers
		matchDetectionsIndependentGreedy(resizedDetectionImage);
	}

	m_bgStatus = BackgroundStatus::WAITING;

	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " detections took: " << trackingDuration << "ms" << std::endl;
}

void ObjectRecognition3d::matchDetectionsIndependentGreedy(const cv::Mat& resizedDetectionImage) {
	m_trackers.reserve(m_trackers.size() + m_bgTrackers.size());
	m_detections.reserve(m_detections.size() + m_bgDetections.size());

	for(size_t i = 0; i < m_bgTrackers.size(); ++i) {
		const auto& bgT = m_bgTrackers[i];
		auto& bgD = m_bgDetections[i];

		assert(bgD.box.width > 0 && bgD.box.height > 0);

		cv::Tracker* fgT = nullptr;
		Detection* fgD = nullptr;
		float maxOverlap = 0;
		for(size_t j = 0; j < m_detections.size(); ++j) {
			auto& d = m_detections[j];
			float overlap = overlapPercentage(bgD.box, d.box);
			if(overlap > maxOverlap) {
				maxOverlap = overlap;
				fgD = &d;
				fgT = m_trackers[j];
			}
		}
		// insert if not overlapping too much with an active detection
		if(maxOverlap < m_overlappingThreshold) {
			bgD.uuid = boost::uuids::random_generator()();
			m_trackers.push_back(bgT);
			m_detections.push_back(bgD);
		} else {	// update if overlapping much
			updateDetection(*fgD, bgD);
			cv::Rect2d box = rect2ImageCoords(resizedDetectionImage, fgD->box);
			fgT->init(resizedDetectionImage, box);
		}
	}

	// detections and trackers moved to foreground or updated foreground
	// delete detections keep trackers, since they can be reused
	m_bgDetections.clear();
}

bool ObjectRecognition3d::getXYZ(const int r, const int c, const float depth,
								 const float cx, const float cy,
								 const float fracfx, const float fracfy,
								 cv::Point3f& point) {
	if(!m_hasRegData) {
		return false;
	}

	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(!std::isfinite(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return false;
	}

	point.x = (c + 0.5f - cx) * fracfx * depth_val;
	point.y = depth_val;
	point.z = -(r + 0.5f - cy) * fracfy * depth_val;

	assert(std::isfinite(point.x) &&
		   std::isfinite(point.y) &&
		   std::isfinite(point.z));
	return true;
}

int32_t ObjectRecognition3d::readNumDetections(const std::vector<tensorflow::Tensor>& outputs) {
	return static_cast<int32_t>(outputs[0].flat<float>().data()[0]);
}

cv::Rect2f ObjectRecognition3d::readDetectionRect(const std::vector<tensorflow::Tensor>& outputs, int32_t idx) {
	float ymin = outputs[1].flat<float>().data()[idx * 4 + 0];
	float xmin = outputs[1].flat<float>().data()[idx * 4 + 1];
	float ymax = outputs[1].flat<float>().data()[idx * 4 + 2];
	float xmax = outputs[1].flat<float>().data()[idx * 4 + 3];

	return cv::Rect2f(xmin, ymin, xmax - xmin, ymax - ymin);
}

float ObjectRecognition3d::readDetectionConfidence(const std::vector<tensorflow::Tensor>& outputs, int32_t idx) {
	return outputs[2].flat<float>().data()[idx];
}

int ObjectRecognition3d::readDetectionType(const std::vector<tensorflow::Tensor>& outputs, int32_t idx) {
	return static_cast<int>(outputs[3].flat<float>().data()[idx]);
}

ObjectRecognition3d::Detection ObjectRecognition3d::readDetection(const std::vector<tensorflow::Tensor>& outputs, int32_t idx) {
	Detection d;
	d.box = readDetectionRect(outputs, idx);
	d.confidence = readDetectionConfidence(outputs, idx);
	d.type = readDetectionType(outputs, idx);
	return d;
}

void ObjectRecognition3d::updateDetection(ObjectRecognition3d::Detection& toUpdate, const ObjectRecognition3d::Detection& data) {
	toUpdate.box = data.box;
	toUpdate.pos = data.pos;
	toUpdate.bboxMax = data.bboxMax;
	toUpdate.bboxMin = data.bboxMin;

	if(toUpdate.confidence < data.confidence) {
		toUpdate.confidence = data.confidence;
		if(toUpdate.type != data.type) {
			toUpdate.type = data.type;
		}
	}
}

void ObjectRecognition3d::updateDetectionBox(Detection& d, const DepthImgType& depth, const cv::Rect2f& box) {
	assert(box.width > 0 && box.height > 0);

	d.box = box;
	d.pos = calcPosition(depth, box);
	calcBBox(d, depth, box);
}

cv::Point3f ObjectRecognition3d::calcPosition(const ObjectRecognition3d::DepthImgType& depthImg,
											  const cv::Rect2f& rect) {

	assert(rect.width >= 0 &&
		   rect.height >= 0);

	cv::Rect2i rrect = rect2ImageCoords(depthImg, rect);
	rrect = clampRect(depthImg, rrect);

	// get array of depths for region of intererst.
	int ymin = rrect.y;
	int ymax = rrect.br().y;
	int xmin = rrect.x;
	int xmax = rrect.br().x;

	// do not use full resolution since depth image is upscaled
	// sample with a 100x100 grid
	// ensure minimal step width of 1
	int yStep = std::max((ymax - ymin) / 100, 1);
	int xStep = std::max((xmax - xmin) / 100, 1);

	// do not always allocat and reserve memory
	m_calcPositionBuffer.clear();
	for(int r = ymin; r < ymax; r+=yStep) {
		for(int c = xmin; c < xmax; c+=xStep) {
			float depth = depthImg(c, r);
			if(std::isfinite(depth) && depth > 0)
				m_calcPositionBuffer.push_back({c, r, depth});
		}
	}

	// get mean of depths between 0.25 and 0.75 quantile
	const auto q1 = m_calcPositionBuffer.size() / 4;
	const auto q2 = m_calcPositionBuffer.size() / 2;
	const auto q3 = q1 + q2;
	std::sort(m_calcPositionBuffer.begin(), m_calcPositionBuffer.end(), [](const ImgDepthPoint& lhs, const ImgDepthPoint& rhs) {
		return std::get<2>(lhs) < std::get<2>(rhs);
	});

	// average points in world space
	const float cx = m_regData.rgb_p.cx;
	const float cy = m_regData.rgb_p.cy;
	const float fracfx = 1/m_regData.rgb_p.fx;
	const float fracfy = 1/m_regData.rgb_p.fy;
	cv::Point3f center(0,0,0);
	cv::Point3f out;
	for(size_t i = q1; i < q3; ++i) {
		int r, c;
		float depth;
		std::tie(c, r, depth) = m_calcPositionBuffer[i];
		if(getXYZ(r, c, depth, cx, cy, fracfx, fracfy, out)) {
#if DEBUG_POS_SAMPLING
			m_currentRGBMarked(c,r) = RGBImgType::Pixel(255, 255, 0);
#endif
			center += out;
		}
	}
	center /= float(q3 - q1);

	return center;
}

void ObjectRecognition3d::calcBBox(ObjectRecognition3d::Detection& d, const ObjectRecognition3d::DepthImgType& depthImage, const cv::Rect2f& box) {
	cv::Rect2f imgBox = rect2ImageCoords(depthImage, box);
	float depth = d.pos.y;
	cv::Point2f tl = imgBox.tl();
	cv::Point2f br = imgBox.br();

	const float cx = m_regData.rgb_p.cx;
	const float cy = m_regData.rgb_p.cy;
	const float fracfx = 1/m_regData.rgb_p.fx;
	const float fracfy = 1/m_regData.rgb_p.fy;

	cv::Point3f tmpBboxMin;
	cv::Point3f tmpBboxMax;
	getXYZ(tl.y, tl.x, depth, cx, cy, fracfx, fracfy, tmpBboxMin);
	getXYZ(br.y, br.x, depth, cx, cy, fracfx, fracfy, tmpBboxMax);

	tmpBboxMax = tmpBboxMax - d.pos;
	tmpBboxMin = tmpBboxMin - d.pos;

	// get max extend along each axis
	float maxX = std::max(std::abs(tmpBboxMin.x), tmpBboxMax.x);
	float maxY = std::max(std::abs(tmpBboxMin.y), tmpBboxMax.y);
	float maxZ = std::max(std::abs(tmpBboxMin.z), tmpBboxMax.z);

	// set extends to max value, use max(x, y) for x and y
	float maxXY = std::max(maxX, maxY);
	d.bboxMax = cv::Point3f(maxXY, maxXY, maxZ);
	d.bboxMin = cv::Point3f(-maxXY, -maxXY, -maxZ);
}

cv::Rect2i ObjectRecognition3d::rect2ImageCoords(const Img<>& image, const cv::Rect2f& rect) {
	return cv::Rect2i(static_cast<int>(rect.x * image.width()),
					  static_cast<int>(rect.y * image.height()),
					  static_cast<int>(rect.width * image.width()),
					  static_cast<int>(rect.height * image.height()));
}

cv::Rect2f ObjectRecognition3d::normalizeRect(const Img<>& image, const cv::Rect2f& rect) {
	return cv::Rect2f(rect.x / image.width(),
					  rect.y / image.height(),
					  rect.width / image.width(),
					  rect.height / image.height());
}

cv::Rect2d ObjectRecognition3d::clampRect(const Img<>& image, const cv::Rect2d& rect) {
	double x0 = std::max(rect.x, 0.0);
	double y0 = std::max(rect.y, 0.0);
	auto br = rect.br();
	double x1 = std::min(br.x, double(image.width()));
	double y1 = std::min(br.y, double(image.height()));
	return cv::Rect2d(x0, y0, x1-x0, y1-y0);
}

float ObjectRecognition3d::overlapPercentage(const cv::Rect2f& r0, const cv::Rect2f& r1) {
	cv::Rect2f unionRect = r0 & r1;
	auto unionArea = unionRect.area();
	return std::max(unionArea / r0.area(), unionArea / r1.area());
}

ObjectRecognition3d::DetectionContainer ObjectRecognition3d::readDetections(
		const std::vector<tensorflow::Tensor>& outputs,
		const Stamped<DepthImgType>& depthImage) {
	DetectionContainer detections;

	int32_t numDetections = readNumDetections(outputs);
	assert(numDetections >= 0);
	detections.reserve(static_cast<size_t>(numDetections));

	for(int32_t i = 0; i < numDetections; ++i) {
		// read out a detection from output tensors
		Detection d = readDetection(outputs, i);

		// update values dependend on new box
		updateDetectionBox(d, depthImage, d.box);

		// save detection to array and post it to channel
		detections.push_back(d);
	}

	return detections;
}

std::vector<ObjectRecognition3d::Detection> ObjectRecognition3d::detect(
		const ObjectRecognition3d::RGBImgType& rgbImage) {
	auto imgSize = rgbImage.size();

	std::vector<Detection> detections;

	tf::Tensor inTensor(tf::DataType::DT_UINT8,
						tf::TensorShape({1, imgSize.height(), imgSize.width(), 3}),
						new Vector2TensorBuffer(
							const_cast<void*>(reinterpret_cast<const void*>(rgbImage.data())),
							static_cast<size_t>(imgSize.width() * imgSize.height()) *
							3 * sizeof(uint8_t)));
	std::vector<std::pair<std::string, tf::Tensor>> inputs = {
		{"image_tensor", inTensor},
	};

	std::vector<tf::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	tf::Status status = m_session->Run(
		inputs, {
			"num_detections",
			"detection_boxes",
			"detection_scores",
			"detection_classes"
		},
		{},
		&outputs);

	if(!status.ok()) {
		std::cout << status.ToString() << "\n";
	}

	int32_t numDetections = readNumDetections(outputs);
	assert(numDetections >= 0);
	detections.reserve(static_cast<size_t>(numDetections));

	for(int32_t i = 0; i < numDetections; ++i) {
		detections.push_back(readDetection(outputs, i));
	}

	return detections;
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
