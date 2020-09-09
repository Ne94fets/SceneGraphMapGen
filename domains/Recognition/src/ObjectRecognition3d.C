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

#include <utils/Time.h>

#include <kinectdatatypes/RGBDOperations.h>
using kinectdatatypes::RGBDOperations;

#include <recognitiondatatypes/TrackerGenerator.h>
using recognitiondatatypes::TrackerGenerator;

#define DEBUG_POS_HIST 0
#define DEBUG_POS_SAMPLING 0
#define DEBUG_BG_FG_TRACKING 0

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

	// reserve memory for size*size points used in calcPosition
	m_calcPositionBuffer.reserve(m_sampleGridSize*m_sampleGridSize);
	m_bgCalcPositionBuffer.reserve(m_sampleGridSize*m_sampleGridSize);
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
	tf::SessionOptions opts;
	opts.config.set_intra_op_parallelism_threads(2);
	opts.config.set_inter_op_parallelism_threads(2);

	tf::Status status = tf::NewSession(opts, &m_session);
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
	m_channelNetDetections = publish<DetectionContainer>("ObjectDetectionNet");
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
	Stamped<ImgPyramid> imgPyramid;
	imgPyramid.frameID = image->frameID;
	imgPyramid.sequenceID = image->sequenceID;
	imgPyramid.timestamp = image->timestamp;

	imgPyramid.value().resize(3);
	image->value().getMat().copyTo(imgPyramid.value()[0]);
	cv::resize(imgPyramid.value()[0], imgPyramid.value()[1], cv::Size(), 0.5, 0.5);
	cv::resize(imgPyramid.value()[1], imgPyramid.value()[2], cv::Size(), 0.5, 0.5);
	m_rgbdQueue.push0(imgPyramid);
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
	while(!m_shutdown && !m_hasRegData) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	std::cout << "ObjectRecognition has registration data. Running in main loop now." << std::endl;

	std::vector<long> durations;
	while(!m_shutdown) {

		// wait for a matching pair
		const auto pair = m_rgbdQueue.getNewestSyncedPair();

		auto startTime = std::chrono::system_clock::now();

		// process the pair
		processPair(pair);

		auto endTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		if(duration > 1000/30) {
			std::cout << "ObjectRec process took: " << duration << "ms" << std::endl;
		}

		durations.push_back(duration);
		float avg(0);
		for(const auto d : durations) {
			avg += float(d);
		}
		avg /= durations.size();
		std::cout << "ObjectRec: Average tracking duration: " << avg << std::endl;
	}
}

void ObjectRecognition3d::backgroundProcess() {
	std::vector<long> durations;
	DetectionContainer newDetections;
	while(!m_shutdown) {
		std::vector<Detection> detections;

		std::unique_lock<std::mutex> imageLock(m_detectionImageMutex);
		m_bgCondition.wait(imageLock,
						   [this]{ return m_bgStatus == BackgroundStatus::WORKING; });

		auto startTime = std::chrono::system_clock::now();

		detections = detect(m_detectionImage);

		// early unlocking
		imageLock.unlock();

		auto endTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		if(duration > 1000/30) {
			std::cout << "Detecting " << detections.size() << " took: " << duration << "ms" << std::endl;
		}

		// set position and bbox, also put into new detections
		auto wChannelNewDetections = m_channelNetDetections.write();
		wChannelNewDetections->sequenceID = m_detectionImage.first.sequenceID;
		DetectionContainer& newDetections = wChannelNewDetections->value();
		newDetections.clear();
		newDetections.reserve(detections.size());
		size_t validCnt = 0;
		for(size_t i = 0; i < detections.size(); ++i) {
			auto& d = detections[i];
			if(d.confidence < m_confidenceThreshold) {
				continue;
			}
			newDetections.push_back(d);
			detections[validCnt++] = d;
		}
		detections.resize(validCnt);

		assert(newDetections.size() == detections.size());

		// swap empty background detections with detections
		std::unique_lock<std::mutex> bgDetectionsLock(m_bgDetectionsMutex);
		std::swap(m_bgDetections, detections);
//		m_bgDetections = detections;
		bgDetectionsLock.unlock();

		durations.push_back(duration);
		float avg(0);
		for(const auto d : durations) {
			avg += float(d);
		}
		avg /= durations.size();
		std::cout << "ObjectRec: Average detection duration: " << avg << std::endl;

		m_bgStatus = BackgroundStatus::DONE;
	}
}

void ObjectRecognition3d::processPair(const ChannelPair& pair) {
	const Stamped<ImgPyramid>& rgbImage = pair.first;
	const Stamped<DepthImgType>& depthImage = pair.second;
	rgbImage[0].getMat().copyTo(m_currentRGBMarked);

	// start a new detection thread if none is running
	startDetection(pair);

	// track last detections
	trackLastDetections(pair);

	// join and clean up old thread if possible and get detections
	trackNewDetections(pair);

	// draw detections to an debug output image
	debugDrawTrackedDetections(pair);

	// post detections
	// new detections will be posted when background process enters done state
	auto wChannelDetections = m_channelDetections.write();
	wChannelDetections->sequenceID = pair.first.sequenceID;
	wChannelDetections->value() = m_detections;

}

void ObjectRecognition3d::startDetection(const ChannelPair& pair) {
	// only start if non is running
	if(m_bgStatus == BackgroundStatus::WAITING) {
		std::unique_lock<std::mutex> imageLock(m_detectionImageMutex);
		m_detectionImage = pair;

		m_bgStatus = BackgroundStatus::WORKING;
		imageLock.unlock();
		m_bgCondition.notify_one();
	}
}

void ObjectRecognition3d::trackLastDetections(const ChannelPair& pair) {
	const Stamped<ImgPyramid>& rgbImage = pair.first;
	const Stamped<DepthImgType>& depthImage = pair.second;
	auto trackingStart = std::chrono::system_clock::now();

	assert(m_detections.size() == m_trackers.size());

	size_t validCnt = 0;
	for(size_t i = 0; i < m_detections.size(); ++i) {
		auto& d = m_detections[i];
		const auto& tracker = m_trackers[i];
		const auto& img = rgbImage.value()[d.pyramidLevel];
		cv::Rect2d box;

		auto trackingStart = std::chrono::system_clock::now();

		bool trackSucess = tracker->update(img, box);
		if(trackSucess && box.height > 0 && box.width > 0) {
			auto trackingEnd = std::chrono::system_clock::now();
			auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
			if(trackingDuration > 1000/30) {
				std::cout << "Tracking #" << i << " took: " << trackingDuration << "ms" << std::endl;
			}

			// clamp box
			box = clampRect(img, box);

			// tracked outside of the image if width and height are smaller than zero after clamping
			if(box.width < 1 || box.height < 1) {
				std::cout << "Tracked object #" << i << " is outside of image" << std::endl;
				continue;
			}

			// update detection
			auto normalized = Detection::normalizeBox(img.size(), box);

			// make sure unexpected growth is not too greate, since not moving that fast
			assert(normalized.area()/d.box.area() <= 2);

			updateDetectionBox(d, pair, normalized, m_calcPositionBuffer);

			// swap detection and tracker to front
			std::swap(m_detections[validCnt], m_detections[i]);
			std::swap(m_trackers[validCnt], m_trackers[i]);
			validCnt++;
		} else {
//			if(d.notVisibleSince < 3) {
//				d.notVisibleSince++;
//				std::swap(m_detections[validCnt], m_detections[i]);
//				std::swap(m_trackers[validCnt], m_trackers[i]);
//				validCnt++;
//			}
			auto trackingEnd = std::chrono::system_clock::now();
			auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
			if(trackingDuration > 1000/30)
				std::cout << "Tracking #" << i << " lost took: " << trackingDuration << "ms" << std::endl;
		}
	}

	// move lost trackers to bg trackers and use them again
	m_bgTrackers.insert(m_bgTrackers.end(), m_trackers.begin() + validCnt, m_trackers.end());

	// post lost detections
	auto wChannelLostDetections = m_channelLostDetections.write();
	wChannelLostDetections->sequenceID = rgbImage.sequenceID;
	DetectionContainer& detectionsLost = wChannelLostDetections->value();
	detectionsLost.clear();
	detectionsLost.insert(detectionsLost.begin(), m_detections.begin() + validCnt, m_detections.end());

	// remove lost detections and thier trackers from active detections
	m_detections.resize(validCnt);
	m_trackers.resize(validCnt);

	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " active detections took: " << trackingDuration << "ms" << std::endl;
}

void ObjectRecognition3d::trackNewDetections(const ChannelPair& pair) {
	const Stamped<ImgPyramid>& rgbImage = pair.first;
	const Stamped<DepthImgType>& depthImage = pair.second;

	auto trackingStart = std::chrono::system_clock::now();

	std::unique_lock<std::mutex> bgDetectionsLock(m_bgDetectionsMutex, std::try_to_lock);

	// return if the background process is still running
	if(m_bgStatus != BackgroundStatus::DONE || !bgDetectionsLock.owns_lock()) {
		return;
	}

	// update new detections to current image
	if(m_bgDetections.empty()) {
		m_bgStatus = BackgroundStatus::WAITING;
		return;
	}

#if DEBUG_BG_FG_TRACKING
	cv::Mat backgroundImage = m_detectionImage.first.value()[0];
	cv::Mat currentImage = rgbImage.value()[0];
#endif

	// add more trackers if needed
	m_bgTrackers.reserve(m_bgDetections.size());
	for(size_t i = m_bgTrackers.size(); i < m_bgDetections.size(); ++i) {
//		m_bgTrackers.push_back(TrackerGenerator::createKCFHOG());
		m_bgTrackers.push_back(cv::TrackerKCF::create());
	}

	assert(m_bgDetections.size() <= m_bgTrackers.size());

	size_t validCnt = 0;
	for(size_t i = 0; i < m_bgDetections.size(); ++i) {

#if DEBUG_BG_FG_TRACKING
		cv::Scalar colors[] = {{255,0,0},
							   {0,255,0},
							   {0,0,255},
							   {255,255,0},
							   {255,0,255},
							   {0,255,255}};
		cv::Scalar& color = colors[i % 6];
#endif
		auto& d = m_bgDetections[i];

		// determine pyramid level from box size
		d.pyramidLevel = calcPyramidLevel(d.box);

		// get detection image
		const auto& detectionImage = m_detectionImage.first.value()[d.pyramidLevel];
		auto shortTermTracker = cv::TrackerMedianFlow::create();

		cv::Rect2d box = Detection::boxOnImage(detectionImage.size(), d.box);

		shortTermTracker->init(detectionImage, box);

#if DEBUG_BG_FG_TRACKING
		{
			cv::Rect2d debug = Detection::boxOnImage(m_currentRGBMarked.size(), d.box);
			cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), debug, cv::Scalar(255, 0, 255), 4);
			cv::rectangle(backgroundImage, debug, color, 4);
		}
#endif

		// get current image
		const auto& img = rgbImage.value()[d.pyramidLevel];

		// try to track to current image
		cv::Rect2d trackedBox;
		if(!shortTermTracker->update(img, trackedBox)) {
			continue;
		}

#if DEBUG_BG_FG_TRACKING
		{
			auto normalized = Detection::normalizeBox(img.size(), trackedBox);
			cv::Rect2d debug = Detection::boxOnImage(m_currentRGBMarked.size(), normalized);
			cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), debug, cv::Scalar(0, 255, 255), 4);
			cv::rectangle(currentImage, debug, color, 4);
		}
#endif

		// clamp box
		trackedBox = clampRect(img, trackedBox);

		// if clamped box too small discard it
		if(trackedBox.width < 1 || trackedBox.height < 1) {
			continue;
		}

		// init faster, better KCF tracker
		m_bgTrackers[i]->init(img, trackedBox);

		// update detection
		auto normalized = Detection::normalizeBox(img.size(), trackedBox);
		updateDetectionBox(d, pair, normalized, m_calcPositionBuffer);

		// swap detection and tracker to front
		std::swap(m_bgDetections[validCnt], m_bgDetections[i]);
		std::swap(m_bgTrackers[validCnt], m_bgTrackers[i]);
		validCnt++;
	}

#if DEBUG_BG_FG_TRACKING
	{
		cv::Mat concatMat(backgroundImage.rows, backgroundImage.cols + currentImage.cols, backgroundImage.type());
		backgroundImage.copyTo(concatMat(cv::Rect(0, 0, backgroundImage.cols, backgroundImage.rows)));
		currentImage.copyTo(concatMat(cv::Rect(backgroundImage.cols, 0, currentImage.cols, currentImage.rows)));
		cv::imshow("Background detect", concatMat);
	}
#endif

	// delete lost trackers
	m_bgDetections.resize(validCnt);
//		m_bgTrackers.resize(validCnt);

	// move new trackers to foreground trackers
	matchDetectionsIndependentGreedy(rgbImage);

	m_bgStatus = BackgroundStatus::WAITING;

	auto trackingEnd = std::chrono::system_clock::now();
	auto trackingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
	if(trackingDuration > 1000/30)
		std::cout << "Tracking " << m_trackers.size() << " background detections took: " << trackingDuration << "ms" << std::endl;
}

void ObjectRecognition3d::matchDetectionsIndependentGreedy(
		const Stamped<ImgPyramid>& rgbImage) {
	m_trackers.reserve(m_trackers.size() + m_bgTrackers.size());
	m_detections.reserve(m_detections.size() + m_bgDetections.size());

	// post new tracked detections
	auto wChannelNewDetections = m_channelNewDetections.write();
	wChannelNewDetections->sequenceID = rgbImage.sequenceID;
	DetectionContainer& detectionsNew = wChannelNewDetections->value();
	detectionsNew.clear();
	detectionsNew.reserve(m_bgDetections.size());

	assert(m_bgDetections.size() <= m_bgTrackers.size());

	size_t notMovedToActiveCnt = 0;
	for(size_t i = 0; i < m_bgDetections.size(); ++i) {
		auto& bgD = m_bgDetections[i];
		const auto& bgT = m_bgTrackers[i];
		const auto& img = rgbImage.value()[bgD.pyramidLevel];

		assert(bgD.box.width > 0 && bgD.box.height > 0);
		assert(m_detections.size() == m_trackers.size());

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
			detectionsNew.push_back(bgD);
		} else {	// update if overlapping much
//			updateDetection(*fgD, bgD);
//			cv::Rect2d box = Detection::boxOnImage(img.size(), fgD->box);
//			fgT->init(img, box);
			std::swap(m_bgTrackers[notMovedToActiveCnt], m_bgTrackers[i]);
			notMovedToActiveCnt++;
		}
	}

	// detections and trackers moved to foreground or updated foreground
	m_bgDetections.clear();
//	m_bgTrackers.resize(notMovedToActiveCnt);
	m_bgTrackers.clear();
}

void ObjectRecognition3d::debugDrawTrackedDetections(const ChannelPair& pair) {
	for(size_t i = 0; i < m_detections.size(); ++i) {
		const auto& d = m_detections[i];
		cv::Scalar color(0, 0, 255);
		if(d.pyramidLevel == 1) {
			color = cv::Scalar(0, 255, 0);
		} else if(d.pyramidLevel == 2) {
			color = cv::Scalar(255, 0, 0);
		}

		cv::Rect resizedRect = Detection::boxOnImage(m_currentRGBMarked.size(), d.box);
		cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), resizedRect, color, 4);

		// draw 3D center into 2D image
		cv::Point2f centerFrom3D;
		if(RGBDOperations::getRowCol(d.pos.x, d.pos.y, d.pos.z,
									 m_regData.rgb_p.cx, m_regData.rgb_p.cy,
									 1/m_regData.rgb_p.fx, 1/m_regData.rgb_p.fy,
									 centerFrom3D.y, centerFrom3D.x)) {
			cv::circle(static_cast<cv::Mat>(m_currentRGBMarked), centerFrom3D, 3, {0, 128, 255}, 2);
		}

		// draw AABB
		drawAABB(m_currentRGBMarked, d, m_regData);

		// draw label
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
		cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), color, -1);
		cv::putText(m_currentRGBMarked.getMat(), topText.str(), textPos,
					cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
		textPos.y += textSize.height + 4;
		std::stringstream textRelPos;
		textRelPos << std::setprecision(2) << std::fixed << d.pos;
		cv::putText(m_currentRGBMarked.getMat(), textRelPos.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);

		std::stringstream bottomText;
		std::string uuidStr = boost::uuids::to_string(d.uuid);
		uuidStr.resize(5);
		bottomText << uuidStr;
		textSize = cv::getTextSize(bottomText.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
		textPos = cv::Point2i(static_cast<int>((d.box.x + d.box.width) * m_currentRGBMarked.width() - textSize.width),
							  static_cast<int>((d.box.y + d.box.height) * m_currentRGBMarked.height() + textSize.height));

		cv::rectangle(static_cast<cv::Mat>(m_currentRGBMarked), cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), color, -1);
		cv::putText(m_currentRGBMarked.getMat(), bottomText.str(), textPos,
					cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
	}

//	m_channelRGBMarked.post(Stamped<RGBImgType>(m_currentRGBMarked,
//												Time::now(),
//												pair.first->sequenceID));
	auto wRGBMarked = m_channelRGBMarked.write();
	wRGBMarked->sequenceID = pair.first.sequenceID;
	wRGBMarked->value() = m_currentRGBMarked.clone();
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
	toUpdate.pyramidLevel = data.pyramidLevel;
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

void ObjectRecognition3d::updateDetectionBox(Detection& d, const ChannelPair& pair, const cv::Rect2f& box, std::vector<ImgDepthPoint>& calcPosBuffer, bool updateColor) {
	assert(box.width > 0 && box.height > 0);

	d.box = box;

	if(updateColor) {
		d.pos = calcPosition(pair, box, calcPosBuffer, &d.color);
	} else {
		d.pos = calcPosition(pair, box, calcPosBuffer);
	}

	calcBBox(d, pair, box);
}

size_t ObjectRecognition3d::calcPyramidLevel(const cv::Rect2f& box) {
	float area = box.area();

	if(area > 0.3*0.3) {
		return 2;
	} else if(area > 0.15*0.15) {
		return 1;
	}

	return 0;
}

cv::Point3f ObjectRecognition3d::calcPosition(const ChannelPair& pair,
											  const cv::Rect2f& rect,
											  std::vector<ImgDepthPoint>& calcPosBuffer,
											  cv::Scalar* color) {

	assert(rect.width >= 0 &&
		   rect.height >= 0);

	const auto& rgbImage = pair.first[0];
	const auto& depthImage = pair.second;

	cv::Rect2i rrect = Detection::boxOnImage(rgbImage.size(), rect);

	// depth image has one extra line at beginning and end
	rrect = clampRect(rgbImage, rrect);
	rrect.y += 1;

	// get array of depths for region of intererst.
	auto br = rrect.br();
	int xmin = rrect.x;
	int ymin = rrect.y;
	int xmax = br.x;
	int ymax = br.y;

	// do not use full resolution since depth image is upscaled
	// sample with a m_sampleGridSize*m_sampleGridSize grid
	// ensure minimal step width of 1
	float xStep = std::max(float(xmax - xmin) / m_sampleGridSize, 1.f);
	float yStep = std::max(float(ymax - ymin) / m_sampleGridSize, 1.f);

	// do not always allocate and reserve memory
	calcPosBuffer.clear();
#if DEBUG_POS_HIST
	size_t binSize = 30; // 3cm
	std::vector<std::vector<ImgDepthPoint>> hist(5000/binSize);
	size_t depthValCnt = 0;
#endif
	for(float fr = ymin; fr < ymax; fr+=yStep) {
		int r = static_cast<int>(fr);
		const float* depthRow = depthImage[r];
		for(float fc = xmin; fc < xmax; fc+=xStep) {
			int c = static_cast<int>(fc);
			float depth = depthRow[c];
			if(std::isfinite(depth) && depth > 0) {
				calcPosBuffer.push_back({c, r, depth});
#if DEBUG_POS_HIST
				size_t bin = static_cast<size_t>(std::round(depth / binSize));
				hist[bin].push_back({c, r, depth});
				depthValCnt++;
#endif
#if DEBUG_POS_SAMPLING
			// only draw tracked detections on current RGB debug
			if(&calcPosBuffer == &m_calcPositionBuffer) {
				assert(0 <= c && c < m_currentRGBMarked.width() &&
					   0 <= r-1 && r-1 < m_currentRGBMarked.height());
				m_currentRGBMarked(c,r-1) = RGBImgType::Pixel(255, 255, 0);
			}
#endif
			}
		}
	}

	// return nan if no depth position is available
	if(calcPosBuffer.size() < 2) {
		return cv::Point3f(std::numeric_limits<float>::quiet_NaN(),
						   std::numeric_limits<float>::quiet_NaN(),
						   std::numeric_limits<float>::quiet_NaN());
	}

	// get mean of depths between 0.25 and 0.75 quantile
	const auto q1 = calcPosBuffer.size() / 4;
	const auto q2 = calcPosBuffer.size() / 2;
	const auto q3 = q1 + q2;
	std::sort(calcPosBuffer.begin(), calcPosBuffer.end(), [](const ImgDepthPoint& lhs, const ImgDepthPoint& rhs) {
		return std::get<2>(lhs) < std::get<2>(rhs);
	});

#if DEBUG_POS_HIST
	int histWidth = 1024;
	int histHeight = 600;
	int binWidth = std::floor(double(histWidth) / hist.size());
	cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255,255,255));
	size_t maxSize = 0;
	for(const auto& bin : hist) {
		if(bin.size() > maxSize) {
			maxSize = bin.size();
		}
	}
	std::cout << "Hist: ";
	for(size_t i = 1; i < hist.size(); ++i) {
		cv::line(histImage,
				 cv::Point(binWidth*(i-1), histHeight - std::round(double(hist[i-1].size()) / depthValCnt * histHeight)),
				cv::Point(binWidth*i, histHeight - std::round(double(hist[i].size()) / depthValCnt * histHeight)),
				cv::Scalar(0,0,0), 2);
		std::cout << "(" << binSize * i << "," << hist[i-1].size() << ") ";
	}
	std::cout << std::endl;
#endif

	// average points in world space
	const float cx = m_regData.rgb_p.cx;
	const float cy = m_regData.rgb_p.cy;
	const float fracfx = 1/m_regData.rgb_p.fx;
	const float fracfy = 1/m_regData.rgb_p.fy;
	cv::Point3f center(0,0,0);
	cv::Point3f out;
	if(!color) {
		for(size_t i = q1; i < q3; ++i) {
			int r, c;
			float depth;
			std::tie(c, r, depth) = calcPosBuffer[i];
			if(!RGBDOperations::getXYZ(r, c, depth, cx, cy, fracfx, fracfy, out)) {
				throw std::runtime_error("Could not calculate 3D position. Bug in code.");
			}
#if DEBUG_POS_SAMPLING
			// only draw tracked detections on current RGB debug
			if(&calcPosBuffer == &m_calcPositionBuffer) {
				assert(0 <= c && c < m_currentRGBMarked.width() &&
					   0 <= r-1 && r-1 < m_currentRGBMarked.height());
				m_currentRGBMarked(c,r-1) = RGBImgType::Pixel(255, 0, 255);
			}
#endif
			center += out;
		}
	} else {
		std::vector<cv::Vec3f> colors;
		colors.reserve(q3-q1);
		for(size_t i = q1; i < q3; ++i) {
			int r, c;
			float depth;
			std::tie(c, r, depth) = calcPosBuffer[i];
			if(!RGBDOperations::getXYZ(r, c, depth, cx, cy, fracfx, fracfy, out)) {
				throw std::runtime_error("Could not calculate 3D position. Bug in code.");
			}
#if DEBUG_POS_SAMPLING
			// only draw tracked detections on current RGB debug
			if(&calcPosBuffer == &m_calcPositionBuffer) {
				assert(0 <= c && c < m_currentRGBMarked.width() &&
					   0 <= r-1 && r-1 < m_currentRGBMarked.height());
				m_currentRGBMarked(c,r-1) = RGBImgType::Pixel(255, 0, 255);
			}
#endif
			center += out;

			// scalar in BGR format as image
			RGBImgType::Pixel pixelColor = rgbImage(c, r-1);
			colors.push_back(cv::Vec3f(pixelColor[0], pixelColor[1], pixelColor[2]));
		}
		std::vector<int> labels;
		std::vector<cv::Vec3f> outColors;
		cv::kmeans(colors, 1, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1), 10, cv::KmeansFlags::KMEANS_RANDOM_CENTERS, outColors);
		*color = outColors[0];
	}
	center /= float(q3 - q1);

#if DEBUG_POS_HIST
	if(&calcPosBuffer == &m_calcPositionBuffer) {
		cv::Size sizeAdd(0,0);// = rrect.size()/4;
		cv::Rect bigRect = clampRect(m_currentRGBMarked, rrect - cv::Point(sizeAdd.width, sizeAdd.height) + sizeAdd*2);
		cv::Mat imgHistRoi = m_currentRGBMarked(bigRect);
//		cv::Mat imgHistPair(std::max(histHeight, imgHistRoi.rows), histWidth + imgHistRoi.cols, CV_8UC3, cv::Scalar(0,0,0));
//		imgHistRoi.copyTo(imgHistPair(cv::Rect(0, 0, imgHistRoi.cols, imgHistRoi.rows)));
//		histImage.copyTo(imgHistPair(cv::Rect(imgHistRoi.cols, 0, histImage.cols, histImage.rows)));
		cv::imshow("Depth Histogram Image", imgHistRoi);
		cv::imwrite("DepthHistImg.png", imgHistRoi);
		cv::imshow("Depth Histogram", histImage);
		size_t i = 0;
//		cv::imshow("Depth Histogramm", imgHistPair);
	}
#endif

	return center;
}

void ObjectRecognition3d::calcBBox(ObjectRecognition3d::Detection& d, const ChannelPair& pair, const cv::Rect2f& box) {
	const auto& rgbImage = pair.first[0];
	cv::Rect2f imgBox = Detection::boxOnImage(rgbImage.size(), box);

	// scale depth to depth image
	float depth = d.pos.y * 1000;

	cv::Point2f tl = imgBox.tl();
	cv::Point2f br = imgBox.br();

	const float cx = m_regData.rgb_p.cx;
	const float cy = m_regData.rgb_p.cy;
	const float fracfx = 1/m_regData.rgb_p.fx;
	const float fracfy = 1/m_regData.rgb_p.fy;

	cv::Point3f tl3;
	cv::Point3f br3;
	RGBDOperations::getXYZ(tl.y, tl.x, depth, cx, cy, fracfx, fracfy, tl3);
	RGBDOperations::getXYZ(br.y, br.x, depth, cx, cy, fracfx, fracfy, br3);

	float minX, maxX, minY, maxY, minZ, maxZ;
	std::tie(minX, maxX) = std::minmax(tl3.x, br3.x);
	std::tie(minY, maxY) = std::minmax(tl3.y, br3.y);
	std::tie(minZ, maxZ) = std::minmax(tl3.z, br3.z);

	cv::Point3f bboxMin(minX, minY, minZ);
	cv::Point3f bboxMax(maxX, maxY, maxZ);

	// get extends relative to pos
	d.bboxMin = bboxMin - d.pos;
	d.bboxMax = bboxMax - d.pos;

	br3 = br3 - d.pos;
	tl3 = tl3 - d.pos;

	// make base of cuboid a square
	float sideX = d.bboxMax.x - d.bboxMin.x;
	float sideY = d.bboxMax.y - d.bboxMin.y;
	if(sideX < sideY) {
		float halfDiff = (sideY - sideX) / 2;
		d.bboxMin.x -= halfDiff;
		d.bboxMax.x += halfDiff;
	} else {
		float halfDiff = (sideX - sideY) / 2;
		d.bboxMin.y -= halfDiff;
		d.bboxMax.y += halfDiff;
	}

}

float ObjectRecognition3d::overlapPercentage(const cv::Rect2f& r0, const cv::Rect2f& r1) {
	cv::Rect2f unionRect = r0 & r1;
	auto unionArea = unionRect.area();
	return std::max(unionArea / r0.area(), unionArea / r1.area());
}

ObjectRecognition3d::DetectionContainer ObjectRecognition3d::readDetections(
		const std::vector<tensorflow::Tensor>& outputs,
		const ChannelPair& pair) {
	DetectionContainer detections;

	int32_t numDetections = readNumDetections(outputs);
	assert(numDetections >= 0);
	detections.reserve(static_cast<size_t>(numDetections));

	for(int32_t i = 0; i < numDetections; ++i) {
		// read out a detection from output tensors
		Detection d = readDetection(outputs, i);

		// update values dependend on new box
		updateDetectionBox(d, pair, d.box, m_bgCalcPositionBuffer, true);

		// save detection to array and post it to channel
		detections.push_back(d);
	}

	return detections;
}

std::vector<ObjectRecognition3d::Detection> ObjectRecognition3d::detect(const ChannelPair& pair) {
	auto& rgbImage = pair.first[0];
	auto imgSize = rgbImage.size();

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

	return readDetections(outputs, pair);
}

cv::Rect2d ObjectRecognition3d::clampRect(const Img<>& image, const cv::Rect2d& rect) {
	double x0 = std::max(rect.x, 0.0);
	double y0 = std::max(rect.y, 0.0);
	auto br = rect.br();
	double x1 = std::min(br.x, static_cast<double>(image.width()));
	double y1 = std::min(br.y, static_cast<double>(image.height()));
	return cv::Rect2d(x0, y0, x1-x0, y1-y0);
}


void ObjectRecognition3d::drawAABB(cv::Mat& img, const Detection& d,
								   const RegistrationData& regData) {
	const float cx = regData.rgb_p.cx;
	const float cy = regData.rgb_p.cy;
	const float fracfx = 1/regData.rgb_p.fx;
	const float fracfy = 1/regData.rgb_p.fy;

	cv::Point3f gMin = d.pos + d.bboxMin;
	cv::Point3f gMax = d.pos + d.bboxMax;

	std::vector<cv::Point3f> boxPoints;
	boxPoints.reserve(8);

	boxPoints.push_back(cv::Point3f(gMin.x, gMin.y, gMin.z));
	boxPoints.push_back(cv::Point3f(gMax.x, gMin.y, gMin.z));
	boxPoints.push_back(cv::Point3f(gMax.x, gMax.y, gMin.z));
	boxPoints.push_back(cv::Point3f(gMin.x, gMax.y, gMin.z));
	boxPoints.push_back(cv::Point3f(gMin.x, gMin.y, gMax.z));
	boxPoints.push_back(cv::Point3f(gMax.x, gMin.y, gMax.z));
	boxPoints.push_back(cv::Point3f(gMax.x, gMax.y, gMax.z));
	boxPoints.push_back(cv::Point3f(gMin.x, gMax.y, gMax.z));

	std::vector<cv::Point2f> imgPoints;
	imgPoints.reserve(8);

	for(const auto& p : boxPoints) {
		cv::Point2f imgP;
		if(!RGBDOperations::getRowCol(p.x, p.y, p.z, cx, cy, fracfx, fracfy, imgP.y, imgP.x)) {
			return;
		}
		imgPoints.push_back(imgP);
	}

	const cv::Scalar color = d.color;//(0, 128, 255);
	const int thickness = 2;

	cv::line(img, imgPoints[0], imgPoints[1], color, thickness);
	cv::line(img, imgPoints[1], imgPoints[2], color, thickness);
	cv::line(img, imgPoints[2], imgPoints[3], color, thickness);
	cv::line(img, imgPoints[3], imgPoints[0], color, thickness);
	cv::line(img, imgPoints[4], imgPoints[5], color, thickness);
	cv::line(img, imgPoints[5], imgPoints[6], color, thickness);
	cv::line(img, imgPoints[6], imgPoints[7], color, thickness);
	cv::line(img, imgPoints[7], imgPoints[4], color, thickness);
	cv::line(img, imgPoints[0], imgPoints[4], color, thickness);
	cv::line(img, imgPoints[1], imgPoints[5], color, thickness);
	cv::line(img, imgPoints[2], imgPoints[6], color, thickness);
	cv::line(img, imgPoints[3], imgPoints[7], color, thickness);
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
