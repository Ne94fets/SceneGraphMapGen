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
	if(m_session)
		delete m_session;

	if(m_net)
		delete m_net;

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

	try {
		m_net = new cv::dnn::DetectionModel("/mnt/hdd/steffen/Projects/GraphMap/external/models/YOLO/yolov3.weights",
											"/mnt/hdd/steffen/Projects/GraphMap/external/models/YOLO/yolov3.cfg");
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	subscribe<RegistrationData>("KinectRegData", &ObjectRecognition3d::onRegistrationData);
	subscribe<RGBImgType>("RGBImageFull", &ObjectRecognition3d::onNewRGBImage);
	subscribe<DepthImgType>("DepthImageFull", &ObjectRecognition3d::onNewDepthImage);

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &ObjectRecognition3d::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
	m_channelRGBMarked = publish<RGBImgType>("RGBImageMarked");
	m_channelDetections = publish<DetectionContainer>("ObjectDetection");

	publishService(*this);

	if(!m_trackThread)
		m_trackThread = new std::thread([this](){ process(); });
	if(!m_bgThread)
		m_bgThread = new std::thread([this](){ backgroundProcess(); });
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
	{
		std::lock_guard<std::mutex> rgbMutex(m_rgbMutex);
		m_rgbQueue.push(image);
	}
}

void ObjectRecognition3d::onNewDepthImage(
		ChannelRead<ObjectRecognition3d::DepthImgType> image) {
	static size_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing depth image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;

	{
		std::lock_guard<std::mutex> depthMutex(m_depthMutex);
		m_depthQueue.push(image);
	}
}

void ObjectRecognition3d::process() {
	while(!m_shutdown) {

		auto startTime = std::chrono::system_clock::now();

		// sync queues and get a matching pair
		const auto optionalPair = getSyncedPair();
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

std::optional<ObjectRecognition3d::ChannelReadPair> ObjectRecognition3d::getSyncedPair() {
	std::optional<ChannelReadPair> pair;

	// get frame number of front image
	{
		std::lock_guard rgbGuard(m_rgbMutex);
		std::lock_guard depthGuard(m_depthMutex);
		if(m_depthQueue.empty() || m_rgbQueue.empty()) {
			return {};
		}

		bool skipping = false;

		while(m_rgbQueue.size() > 1 && m_depthQueue.size() > 1) {
			// pop rgb images before next depth image
			while(!m_rgbQueue.empty() && m_rgbQueue.front()->sequenceID < m_depthQueue.front()->sequenceID) {
				std::cout << "No matching depth image. Dropping RGB image. FrameNumber: " << m_rgbQueue.front()->sequenceID << std::endl;
				m_rgbQueue.pop();
			}
			// pop depth images before next rgb image
			while(!m_depthQueue.empty() && m_depthQueue.front()->sequenceID < m_rgbQueue.front()->sequenceID) {
				std::cout << "No matching rgb image. Dropping depth image. FrameNumber: " << m_depthQueue.front()->sequenceID << std::endl;
				m_depthQueue.pop();
			}

			pair = std::make_pair(m_rgbQueue.front(), m_depthQueue.front());

			// rgb and depth image should be synchronized and have same frame number now
			assert(pair->first->sequenceID == pair->second->sequenceID);

			m_rgbQueue.pop();
			m_depthQueue.pop();
			if(skipping) {
				std::cout << "Skipping frames: " << pair->first->sequenceID << std::endl;
			}
			skipping = true;
		}
	}

	return pair;
}

void ObjectRecognition3d::processPair(const ChannelReadPair& pair) {
	const Stamped<RGBImgType>& rgbImage = pair.first;
	const Stamped<DepthImgType>& depthImage = pair.second;
	Stamped<RGBImgType> rgbSmall;
	rgbSmall.sequenceID = pair.first->sequenceID;
	cv::resize(rgbImage, rgbSmall, cv::Size(rgbImage.width()/4, rgbImage.height()/4));

	// start a new detection thread if none is running
	startDetection(pair.first);

	// track last detections
	trackLastDetections(rgbSmall, depthImage);

	// join and clean up old thread if possible and get detections
	trackNewDetections(rgbSmall, depthImage);

	// draw detections to an debug output image
	auto drawStart = std::chrono::system_clock::now();

	RGBImgType outRGB;
	rgbImage.getMat().copyTo(outRGB);

	for(const auto& d : m_detections) {
		cv::Rect resizedRect = rect2ImageCoords(outRGB, d.box);
		cv::rectangle(static_cast<cv::Mat>(outRGB), resizedRect, cv::Scalar(0, 0, 255), 4);

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
		auto textPos = cv::Point2i(static_cast<int>(d.box.x * outRGB.width()),
								   static_cast<int>(d.box.y * outRGB.height()));
		cv::rectangle(static_cast<cv::Mat>(outRGB), cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), cv::Scalar(0, 0, 255), -1);
		cv::putText(outRGB.getMat(), topText.str(), textPos,
					cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
		textPos.y += textSize.height + 4;
		std::stringstream textRelPos;
		textRelPos << d.pos;
		cv::putText(outRGB.getMat(), textRelPos.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255), thickness);
	}

	auto drawEnd = std::chrono::system_clock::now();
	auto drawDuration = std::chrono::duration_cast<std::chrono::milliseconds>(drawEnd - drawStart).count();
	if(drawDuration > 1000/30)
		std::cout << "Draw detection took: " << drawDuration << "ms" << std::endl;

	auto wRGBMarked = m_channelRGBMarked.write();
	wRGBMarked->sequenceID = pair.first->sequenceID;
	wRGBMarked->value() = outRGB;

	// post detections
	auto wChannelDetections = m_channelDetections.write();
	wChannelDetections->sequenceID = pair.first->sequenceID;
	wChannelDetections->value() = m_detections;
}

void ObjectRecognition3d::startDetection(const ChannelRead<RGBImgType>& rgbImage) {
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

	std::stack<size_t> lostIndices;
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
				lostIndices.push(i);
				std::cout << "Tracked object #" << i << " is outside of image" << std::endl;
				continue;
			}

			// update detection
			auto normalized = normalizeRect(rgbImage, box);
			auto& d = m_detections[i];
			d.frameNumber = rgbImage.sequenceID;
			d.box = normalized;
			d.pos = calcPosition(depthImage, normalized);
		} else {
			lostIndices.push(i);
			auto trackingEnd = std::chrono::system_clock::now();
			auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
			if(trackingDuration > 1000/30)
				std::cout << "Tracking #" << i << " lost took: " << trackingDuration << "ms" << std::endl;
		}
	}

	// move lost trackers to bg trackers and use them again
	m_bgTrackers.reserve(m_bgTrackers.size() + lostIndices.size());
	while(!lostIndices.empty()) {
		size_t idx = lostIndices.top();
		lostIndices.pop();

		m_bgTrackers.push_back(m_trackers[idx]);

		long offset = static_cast<long>(idx);
		m_trackers.erase(m_trackers.begin() + offset);
		m_detections.erase(m_detections.begin() + offset);
	}


	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " detections took: " << trackingDuration << "ms" << std::endl;
}

void ObjectRecognition3d::trackNewDetections(const Stamped<RGBImgType>& rgbImage,
											 const Stamped<DepthImgType>& depthImage) {
	auto trackingStart = std::chrono::system_clock::now();

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
		cv::resize(*m_detectionImage, resizedDetectionImage, rgbImage.size());

		std::stack<size_t> lostIndices;
		for(size_t i = 0; i < m_bgDetections.size(); ++i) {
			auto& d = m_bgDetections[i];
			cv::Rect2d box = rect2ImageCoords(resizedDetectionImage, d.box);
			m_bgTrackers[i]->init(resizedDetectionImage, box);

			// try to track to current image
			if(m_bgTrackers[i]->update(rgbImage, box)) {
				// clamp box
				box = clampRect(rgbImage, box);

				if(box.width < 0 || box.height < 0) {
					lostIndices.push(i);
					continue;
				}

				// update detection
				auto normalized = normalizeRect(rgbImage, box);
				d.frameNumber = rgbImage.sequenceID;
				d.box = normalized;
				d.pos = calcPosition(depthImage, normalized);
			} else {
				lostIndices.push(i);
			}
		}

		// delete lost trackers
		while(!lostIndices.empty()) {
			long offset = static_cast<long>(lostIndices.top());
			lostIndices.pop();
			m_bgDetections.erase(m_bgDetections.begin() + offset);
			m_bgTrackers.erase(m_bgTrackers.begin() + offset);
		}
		//lostIndices empty now

		// move new trackers to foreground trackers
		m_trackers.reserve(m_trackers.size() + m_bgTrackers.size());
		m_detections.reserve(m_detections.size() + m_bgDetections.size());
		for(size_t i = 0; i < m_bgTrackers.size(); ++i) {
			const auto& bgT = m_bgTrackers[i];
			const auto& bgD = m_bgDetections[i];
			float maxOverlap = 0;
			for(auto it = m_detections.begin(); it != m_detections.end(); ++it) {
				const auto& d = *it;
				float overlap = overlapPercentage(bgD.box, d.box);
				if(overlap > maxOverlap)
					maxOverlap = overlap;
			}
			// insert if not overlapping too much with an active detection
			if(maxOverlap < m_overlappingThreshold) {
				m_trackers.push_back(bgT);
				m_detections.push_back(bgD);
				lostIndices.push(i);
			}
		}

		// delete lost trackers since data was moved to foreground arrays
		while(!lostIndices.empty()) {
			long offset = static_cast<long>(lostIndices.top());
			lostIndices.pop();
			m_bgDetections.erase(m_bgDetections.begin() + offset);
			m_bgTrackers.erase(m_bgTrackers.begin() + offset);
		}
		m_bgDetections.clear();
	}

	m_bgStatus = BackgroundStatus::WAITING;

	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " detections took: " << trackingDuration << "ms" << std::endl;
}

cv::Point3f ObjectRecognition3d::getXYZ(const int r, const int c, const float depth,
										const float cx, const float cy,
										const float fracfx, const float fracfy) {
	const float bad_point = std::numeric_limits<float>::quiet_NaN();
	if(!m_hasRegData)
		return cv::Point3f(bad_point, bad_point, bad_point);

	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(isnan(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return cv::Point3f(bad_point, bad_point, bad_point);
	} else {
		float x = (c + 0.5f - cx) * fracfx * depth_val;
		float y = (r + 0.5f - cy) * fracfy * depth_val;
		float z = depth_val;
		return cv::Point3f(x, y, z);
	}
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

cv::Point3f ObjectRecognition3d::calcPosition(const ObjectRecognition3d::DepthImgType& depthImg,
											  const cv::Rect2f& rect) {

	assert(rect.width >= 0 &&
		   rect.height >= 0);

	// get array of depths for region of intererst.
	int ymin = static_cast<int>(rect.y * depthImg.height());
	int ymax = static_cast<int>((rect.y + rect.height) * depthImg.height());
	int xmin = static_cast<int>(rect.x * depthImg.width());
	int xmax = static_cast<int>((rect.x + rect.width) * depthImg.width());

	// do not use full resolution since depth image is upscaled
	// sample with a 100x100 grid
	// ensure minimal step width of 1
	int yStep = std::max((ymax - ymin) / 100, 1);
	int xStep = std::max((xmax - xmin) / 100, 1);

	// do not always allocat and reserve memory
	m_calcPositionBuffer.clear();
	for(int y = ymin; y < ymax; y+=yStep) {
		for(int x = xmin; x < xmax; x+=xStep) {
			float depth = depthImg(x, y);
			if(std::isfinite(depth) && depth > 0)
				m_calcPositionBuffer.push_back({x, y, depth});
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
	for(size_t i = q1; i < q3; ++i) {
		int r, c;
		float depth;
		std::tie(r, c, depth) = m_calcPositionBuffer[i];
		const auto pInWorldCoords = getXYZ(r, c, depth,
										   cx, cy, fracfx, fracfy);
		assert(std::isfinite(pInWorldCoords.x) &&
			   std::isfinite(pInWorldCoords.y) &&
			   std::isfinite(pInWorldCoords.z));
		center += pInWorldCoords;
	}
	center /= float(q3 - q1);

	return center;
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

		// give it the frame number of the current image
		d.frameNumber = depthImage.sequenceID;

		// calculate the center position of the dectection
		d.pos = calcPosition(depthImage, d.box);

		// save detection to array and post it to channel
		detections.push_back(d);
	}

	return detections;
}

std::vector<ObjectRecognition3d::Detection> ObjectRecognition3d::detect(
		const ObjectRecognition3d::RGBImgType& rgbImage) {
	auto imgSize = rgbImage.size();
	cv::Mat detMat;
	cv::Size detSize(rgbImage.width()/2, rgbImage.width()/2);
	cv::resize(rgbImage, detMat, detSize);

	std::vector<Detection> detections;

	// use opencv net if there is one
	if(m_net) {
		std::vector<int> classIds;
		std::vector<float> conf;
		std::vector<cv::Rect> boxes;
		m_net->setInputSize(detSize);
		m_net->detect(detMat, classIds, conf, boxes);

		for(size_t i = 0; i < classIds.size(); ++i) {
			Detection d;
			d.type = classIds[i] + 1;
			d.confidence = conf[i];
			d.box = normalizeRect(detMat, boxes[i]);
			detections.push_back(d);
		}
	} else {	// use tensorflow
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

		for(int32_t i = 0; i < numDetections; ++i)
			detections.push_back(readDetection(outputs, i));
	}

	return detections;
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
