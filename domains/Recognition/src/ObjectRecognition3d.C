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

#include <exception>
#include <iomanip>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <opencv2/imgproc.hpp>

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
}

ObjectRecognition3d::~ObjectRecognition3d() {
	if(m_session)
		delete m_session;
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

	publishService(*this);
}

void ObjectRecognition3d::onRegistrationData(ChannelRead<ObjectRecognition3d::RegistrationData> data) {
	if(m_hasRegData)
		return;

	m_regData = *data;
	m_hasRegData = true;
}

void ObjectRecognition3d::onNewRGBImage(ChannelRead<ObjectRecognition3d::RGBImgType> image) {
	{
		std::lock_guard<std::mutex> rgbMutex(m_rgbMutex);
		m_rgbQueue.push(*image);
	}
	process();
}

void ObjectRecognition3d::onNewDepthImage(ChannelRead<ObjectRecognition3d::DepthImgType> image) {
	{
		std::lock_guard<std::mutex> depthMutex(m_depthMutex);
		m_depthQueue.push(*image);
	}
	process();
}

cv::Point3f ObjectRecognition3d::getXYZ(int r, int c, float depth) {
	const float bad_point = std::numeric_limits<float>::quiet_NaN();
	if(!m_hasRegData)
		return cv::Point3f(bad_point, bad_point, bad_point);

	const float cx(m_regData.depth_p.cx), cy(m_regData.depth_p.cy);
	const float fx(1 / m_regData.depth_p.fx), fy(1 / m_regData.depth_p.fy);
	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(isnan(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return cv::Point3f(bad_point, bad_point, bad_point);
	} else {
		float x = (c + 0.5f - cx) * fx * depth_val;
		float y = (r + 0.5f - cy) * fy * depth_val;
		float z = depth_val;
		return cv::Point3f(x, y, z);
	}
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
	typedef std::tuple<int, int, float> ImgDepthPoint;
	// get array of depths for region of intererst.
	std::vector<ImgDepthPoint> roiPoints;
	int ymin = static_cast<int>(rect.y * depthImg.height());
	int ymax = static_cast<int>((rect.y + rect.height) * depthImg.height());
	int xmin = static_cast<int>(rect.x * depthImg.width());
	int xmax = static_cast<int>((rect.x + rect.width) * depthImg.width());

	// do not use full resolution since depth image is upscaled
	for(int y = ymin; y < ymax; y+=2) {
		for(int x = xmin; x < xmax; x+=2) {
			float depth = depthImg(x, y);
			if(!std::isnan(depth))
				roiPoints.push_back({x, y, depth});
		}
	}

	// get mean of depths between 0.25 and 0.75 quantile
	const auto q1 = roiPoints.size() / 4;
	const auto q2 = roiPoints.size() / 2;
	const auto q3 = q1 + q2;
	std::sort(roiPoints.begin(), roiPoints.end(), [](const ImgDepthPoint& lhs, const ImgDepthPoint& rhs) {
		return std::get<2>(lhs) < std::get<2>(rhs);
	});

	// average points in world space
	cv::Point3f center(0,0,0);
	for(size_t i = q1; i < q3; ++i) {
		int r, c;
		float depth;
		std::tie(r, c, depth) = roiPoints[i];
		const auto pInWorldCoords = getXYZ(r, c, depth);	// TODO: getXYZ is from small depth image
		center += pInWorldCoords;
	}
	center /= float(q3 - q1);

	return center;
}

void ObjectRecognition3d::syncQueues() {
	{
		// lock queue before iterating over it
		std::lock_guard<std::mutex> rgbGuard(m_rgbMutex);

		// pop rgb images before next depth image
		while(!m_rgbQueue.empty() && m_rgbQueue.front().frameNumber() < m_depthQueue.front().frameNumber()) {
			std::cout << "Dropping RGB image. FrameNumber: " << m_rgbQueue.front().frameNumber() << std::endl;
			m_rgbQueue.pop();
		}
	}

	if(m_rgbQueue.empty())
		return;

	{
		// lock queue before iterating over it
		std::lock_guard<std::mutex> depthGuard(m_depthMutex);

		// pop depth images before next rgb image
		while(!m_depthQueue.empty() && m_depthQueue.front().frameNumber() < m_rgbQueue.front().frameNumber()) {
			std::cout << "Dropping depth image. FrameNumber: " << m_depthQueue.front().frameNumber() << std::endl;
			m_depthQueue.pop();
		}
	}
}

std::vector<tensorflow::Tensor> ObjectRecognition3d::detect(const ObjectRecognition3d::RGBImgType& rgbImage) {
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

	return outputs;
}

void ObjectRecognition3d::process() {
	std::lock_guard<std::mutex> guard(m_processingMutex);

	auto startTime = std::chrono::system_clock::now();
	while(!m_rgbQueue.empty() && !m_depthQueue.empty()) {
		auto tmpStartTime = std::chrono::system_clock::now();

		// sync queues
		syncQueues();

		// return if any of the queues is empty
		if(m_rgbQueue.empty() || m_depthQueue.empty())
			return;

		// rgb and depth image should be synchronized and have same frame number now
		const auto& rgbImage = m_rgbQueue.front();
		const auto& depthImage = m_depthQueue.front();
		assert(rgbImage.frameNumber() == depthImage.frameNumber());

		// detect objects on rgb image
		auto outputs = detect(rgbImage);

		auto tmpEndTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tmpEndTime - tmpStartTime).count();
		tmpStartTime = tmpEndTime;
		std::cout << "Detecting took: " << duration << "ms" << std::endl;

		if(outputs.empty())
			return;

		// process detections
		int32_t numDetections = static_cast<int32_t>(outputs[0].flat<float>().data()[0]);

		RGBImgType outRGB;
		rgbImage.getMat().copyTo(outRGB);
		outRGB.frameNumber() = rgbImage.frameNumber();

		DetectionContainer detections;
		assert(numDetections >= 0);
		detections.reserve(static_cast<size_t>(numDetections));

		long channelDuration = 0;
		long drawDuration = 0;
		for(int32_t i = 0; i < numDetections; ++i) {
			auto channelStart = std::chrono::system_clock::now();
			// read out a detection from output tensors
			Detection d = readDetection(outputs, i);

			// give it the frame number of the current image
			d.frameNumber = rgbImage.frameNumber();

			// calculate the center position of the dectection
			d.pos = calcPosition(depthImage, d.box);

			// save detection to array and post it to channel
			detections.push_back(d);
//			auto wChannelDetection = m_channelDetections.write();
//			wChannelDetection->value() = d;
//			m_channelDetection.post(d);
			auto channelEnd = std::chrono::system_clock::now();
			channelDuration += std::chrono::duration_cast<std::chrono::milliseconds>(channelEnd - channelStart).count();

			auto drawStart = std::chrono::system_clock::now();
			// draw detection to an debug output image
			cv::Rect resizedRect(static_cast<int>(d.box.x * outRGB.width()),
								 static_cast<int>(d.box.y * outRGB.height()),
								 static_cast<int>(d.box.width * outRGB.width()),
								 static_cast<int>(d.box.height * outRGB.height()));
			cv::rectangle(outRGB, resizedRect, cv::Scalar(0, 0, 255), 4);

			std::stringstream topText;
			topText << Detection::getTypeName(d.type) << ": " << std::setprecision(0) << std::fixed << d.confidence * 100;
			int baseline = 0;
			double fontScale = 1;
			int thickness = 2;
			auto textSize = cv::getTextSize(topText.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
			auto textPos = cv::Point2i(static_cast<int>(d.box.x * outRGB.width()),
									   static_cast<int>(d.box.y * outRGB.height()));
			cv::rectangle(outRGB, cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), cv::Scalar(0, 0, 255), -1);
			cv::putText(outRGB.getMat(), topText.str(), textPos,
						cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
			textPos.y += textSize.height + 4;
			std::stringstream textRelPos;
			textRelPos << d.pos;
			cv::putText(outRGB.getMat(), textRelPos.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255), thickness);
			auto drawEnd = std::chrono::system_clock::now();
			drawDuration += std::chrono::duration_cast<std::chrono::milliseconds>(drawEnd - drawStart).count();
		}

		// post detections
		auto wChannelDetections = m_channelDetections.write();
		wChannelDetections->value() = detections;
//		m_channelDetections.post(detections);

		tmpEndTime = std::chrono::system_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(tmpEndTime - tmpStartTime).count();
		tmpStartTime = tmpEndTime;
		std::cout << "Reading detection took: " << channelDuration << "ms" << std::endl;
		std::cout << "Draw detection took: " << drawDuration << "ms" << std::endl;
		std::cout << "Processing Detections took: " << duration << "ms" << std::endl;

		auto wRGBMarked = m_channelRGBMarked.write();
		wRGBMarked->value() = outRGB;

		m_lastDetections = detections;

		{
			std::lock_guard<std::mutex> rgbGuard(m_rgbMutex);
			m_rgbQueue.pop();
		}
		{
			std::lock_guard<std::mutex> depthGuard(m_depthMutex);
			m_depthQueue.pop();
		}

		tmpEndTime = std::chrono::system_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(tmpEndTime - tmpStartTime).count();
		tmpStartTime = tmpEndTime;
		std::cout << "Postprocessing took: " << duration << "ms" << std::endl;
	}

	auto endTime = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "Detectionprocess took: " << duration << "ms" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
