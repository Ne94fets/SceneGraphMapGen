 
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

#ifndef RECOGNITION_OBJECTRECKOGNITION3D_H
#define RECOGNITION_OBJECTRECKOGNITION3D_H

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/MicroUnit.h>

#include <optional>
#include <mutex>
#include <thread>
#include <utility>

#include <recognitiondatatypes/Detection.h>
#include <kinectdatatypes/Types.h>
#include <kinectdatatypes/RGBDQueue.h>

#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>

using namespace mira;

// forward declarations
namespace tensorflow {
	class Session;
	class Tensor;
} // namespace tensorflow

namespace tf = tensorflow;

namespace recognition {

///////////////////////////////////////////////////////////////////////////////

/**
 * Recognizes object in RGB images and locates them in 3D space
 */
class ObjectRecognition3d : public MicroUnit {
	MIRA_OBJECT(ObjectRecognition3d)

public:
	typedef kinectdatatypes::RegistrationData	RegistrationData;

	typedef Img<uint8_t, 3>	RGBImgType;
	typedef Img<float, 1>	DepthImgType;

	typedef recognitiondatatypes::Detection				Detection;
	typedef recognitiondatatypes::DetectionContainer	DetectionContainer;

	typedef kinectdatatypes::RGBDQueue<Stamped<RGBImgType>, Stamped<DepthImgType>>	SyncQueueType;
	typedef typename SyncQueueType::ChannelPair										ChannelPair;

	typedef std::tuple<int, int, float>	ImgDepthPoint;

public:
	ObjectRecognition3d();
	virtual ~ObjectRecognition3d();

	template<typename Reflector>
	void reflect(Reflector& r) {
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&ObjectRecognition3d::setParam2,this), "Second parameter with setter");
	}

protected:
	virtual void initialize();

private:
	void onRegistrationData(ChannelRead<RegistrationData> data);
	void onNewRGBImage(ChannelRead<RGBImgType> image);
	void onNewDepthImage(ChannelRead<DepthImgType> image);

	void process();
	void backgroundProcess();

	void processPair(const ChannelPair& pair);
	void startDetection(const Stamped<RGBImgType>& rgbImage);
	void trackLastDetections(const Stamped<RGBImgType>& rgbImage,
							 const Stamped<DepthImgType>& depthImage);
	void trackNewDetections(const Stamped<RGBImgType>& rgbImage,
							const Stamped<DepthImgType>& depthImage);

	void matchDetectionsIndependentGreedy(const cv::Mat& resizedDetectionImage);

	bool getXYZ(const int r, const int c, const float depth,
				const float cx, const float cy,
				const float fracfx, const float fracfy,
				cv::Point3f& point);

	int32_t		readNumDetections(const std::vector<tf::Tensor>& outputs);
	cv::Rect2f	readDetectionRect(const std::vector<tf::Tensor>& outputs, int32_t idx);
	float		readDetectionConfidence(const std::vector<tf::Tensor>& outputs, int32_t idx);
	int			readDetectionType(const std::vector<tf::Tensor>& outputs, int32_t idx);
	Detection	readDetection(const std::vector<tf::Tensor>& outputs, int32_t idx);

	void	updateDetection(Detection& toUpdate, const Detection& data);
	void	updateDetectionBox(Detection& d, const DepthImgType& depth, const cv::Rect2f& box);

	cv::Point3f			calcPosition(const DepthImgType& depthImg, const cv::Rect2f& rect);
	void				calcBBox(Detection& d, const DepthImgType& depthImage, const cv::Rect2f& box);
	cv::Rect2i			rect2ImageCoords(const Img<>& image, const cv::Rect2f& rect);
	cv::Rect2f			normalizeRect(const Img<>& image, const cv::Rect2f& rect);
	cv::Rect2d			clampRect(const Img<>& image, const cv::Rect2d& rect);
	float				overlapPercentage(const cv::Rect2f& r0, const cv::Rect2f& r1);
	DetectionContainer	readDetections(const std::vector<tf::Tensor>& outputs,
									   const Stamped<DepthImgType>& depthImage);

	std::vector<Detection> detect(const RGBImgType& rgbImage);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:
	enum class BackgroundStatus {
		WORKING,
		DONE,
		WAITING
	};

	volatile bool	m_shutdown = false;

	float	m_overlappingThreshold = 0.5f;
	float	m_confidenceThreshold = 0.5f;

	RGBImgType						m_currentRGBMarked;
	Channel<RGBImgType>				m_channelRGBMarked;
	Channel<DetectionContainer>		m_channelDetections;
	Channel<DetectionContainer>		m_channelNewDetections;
	Channel<DetectionContainer>		m_channelLostDetections;

	SyncQueueType	m_rgbdQueue;

	tf::Session*	m_session = nullptr;

	RegistrationData	m_regData;
	bool				m_hasRegData = false;

	std::mutex	m_processingMutex;

	std::thread*	m_trackThread = nullptr;
	std::thread*	m_bgThread = nullptr;

	BackgroundStatus		m_bgStatus = BackgroundStatus::WAITING;
	Stamped<RGBImgType>		m_detectionImage;
	std::mutex				m_detectionImageMutex;

	std::vector<ImgDepthPoint>			m_calcPositionBuffer;

	std::vector<Detection>				m_bgDetections;
	std::vector<cv::Ptr<cv::Tracker>>	m_bgTrackers;

	DetectionContainer					m_detections;
	DetectionContainer					m_detectionsNew;
	DetectionContainer					m_detectionsLost;
	std::vector<cv::Ptr<cv::Tracker>>	m_trackers;
};

} // namespace recognition

#endif // RECOGNITION_OBJECTRECKOGNITION3D_H
