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
 * @file PointCloudRegistration.C
 *    Creates a combinded Point Cloud and extracts all kinds of data.
 *
 * @author Steffen Kastner
 * @date   2020/05/14
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/MicroUnit.h>

#include <thread>
#include <chrono>

#include <pcl/point_cloud.h>

#include <pcl/memory.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <opencv2/features2d.hpp>

#include <kinectdatatypes/Types.h>
#include <kinectdatatypes/RGBDQueue.h>

using namespace mira;
using namespace pcl;

namespace recognition {

///////////////////////////////////////////////////////////////////////////////

/**
 * Creates a combinded Point Cloud and extracts all kinds of data.
 */
class PointCloudRegistration : public MicroUnit
{
MIRA_OBJECT(PointCloudRegistration)

public:
	typedef kinectdatatypes::RegistrationData	KinectRegistrationData;

	typedef Img<uint8_t, 3>	RGBImgType;
	typedef Img<float, 1>	DepthImgType;

	typedef PointXYZ				PointType;
	typedef Eigen::Matrix4f			TransformType;
	typedef PointCloud<PointType>	PointCloudType;

	typedef std::pair<ChannelRead<RGBImgType>, ChannelRead<DepthImgType>>	ChannelReadPair;

public:

	PointCloudRegistration();
	~PointCloudRegistration();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&PointCloudRegistration::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &PointCloudRegistration::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

private:

	void onKinectRegistrationData(ChannelRead<KinectRegistrationData> data);
	void onDepthImage(ChannelRead<DepthImgType> image);
	void onRGBImage(ChannelRead<RGBImgType> image);
	void process();
	void processPair(const ChannelReadPair& pair);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

	void pairAlignSrc2Target(const PointCloudType::Ptr cloud_src,
			const PointCloudType::Ptr cloud_tgt,
			Eigen::Matrix4f& target2Src,
			bool downsample = false);

	PointCloudType::Ptr pair2Cloud(const ChannelReadPair& pair);
	bool getXYZ(int r, int c, float depth, PointType& out);

private:
	volatile bool m_shutdown = false;

	float	m_cx, m_cy;
	float	m_fx, m_fy;
	bool	m_hasKinectRegData = false;

	kinectdatatypes::RGBDQueue	m_rgbdQueue;

	cv::Ptr<cv::FastFeatureDetector>	m_fastDetector;
	PointCloudType::Ptr					m_lastCloud;

	std::thread*	m_thread = nullptr;

	TransformType		m_globalTransform = TransformType::Identity();

	Channel<TransformType>	m_channelGlobalTransfrom;
	Channel<TransformType>	m_channelLocalTransform;
};

// Define a new point representation for < x, y, z >
//class PointRep : public pcl::PointRepresentation<Eigen::Vector3f> {
//  using pcl::PointRepresentation<Eigen::Vector3f>::nr_dimensions_;

//public:
//	PointRep();

//	// Override the copyToFloatArray method to define our feature vector
//	virtual void copyToFloatArray (const Eigen::Vector3f& p, float* out) const;
//};

//PointRep::PointRep() {
//	// Define the number of dimensions
//	nr_dimensions_ = 3;
//}

//void PointRep::copyToFloatArray(const Eigen::Vector3f& p, float* out) const {
//	// < x, y, z >
//	out[0] = p.x();
//	out[1] = p.y();
//	out[2] = p.z();
//}

///////////////////////////////////////////////////////////////////////////////

PointCloudRegistration::PointCloudRegistration() {
	// TODO: further initialization of members, etc.
	m_fastDetector = cv::FastFeatureDetector::create();
}

PointCloudRegistration::~PointCloudRegistration() {
	m_shutdown = true;

	if(m_thread) {
		m_thread->join();
		delete m_thread;
	}
}

void PointCloudRegistration::initialize() {
	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &PointCloudRegistration::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
	m_channelGlobalTransfrom = publish<TransformType>("PCGlobalTransform");
	m_channelLocalTransform = publish<TransformType>("PCLocalTransform");

	subscribe<KinectRegistrationData>("KinectRegData", &PointCloudRegistration::onKinectRegistrationData);
	subscribe<RGBImgType>("RGBImageFull", &PointCloudRegistration::onRGBImage);
	subscribe<DepthImgType>("DepthImage", &PointCloudRegistration::onDepthImage);

	if(!m_thread) {
		m_thread = new std::thread([this](){ process(); });
	}
}

void PointCloudRegistration::onKinectRegistrationData(ChannelRead<KinectRegistrationData> data) {
	if(m_hasKinectRegData)
		return;

//	m_cx = data->depth_p.cx;
//	m_cy = data->depth_p.cy;
//	m_fx = 1 / data->depth_p.fx;
//	m_fy = 1 / data->depth_p.fy;
	m_cx = data->rgb_p.cx;
	m_cy = data->rgb_p.cy;
	m_fx = 1 / data->rgb_p.fx;
	m_fy = 1 / data->rgb_p.fy;
	m_hasKinectRegData = true;
}

void PointCloudRegistration::onDepthImage(ChannelRead<DepthImgType> image) {
	static size_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing depth image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	m_rgbdQueue.push(image);
}

void PointCloudRegistration::onRGBImage(ChannelRead<PointCloudRegistration::RGBImgType> image) {
	static uint32_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing color image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	m_rgbdQueue.push(image);
}

void PointCloudRegistration::process() {
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

void PointCloudRegistration::processPair(const ChannelReadPair& pair) {
	if(!m_lastCloud) {
		m_lastCloud = pair2Cloud(pair);
		return;
	}

	Eigen::Matrix4f transform;
	auto newCloud = pair2Cloud(pair);
	auto startTime = std::chrono::system_clock::now();
	pairAlignSrc2Target(newCloud, m_lastCloud, transform, false);
	auto endTime = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
	std::cout << "PCL align took: " << duration << "ms" << std::endl;

	// get current time
	auto timestamp = Time::now();

	// publish local transfrom
	m_channelLocalTransform.post(Stamped<TransformType>(transform, timestamp, pair.first->sequenceID));

	// accumulate transform
	m_globalTransform = transform * m_globalTransform;
	m_channelGlobalTransfrom.post(Stamped<TransformType>(m_globalTransform, timestamp, pair.first->sequenceID));

	m_lastCloud = newCloud;
}

void PointCloudRegistration::pairAlignSrc2Target(
		const PointCloudType::Ptr cloud_src,
		const PointCloudType::Ptr cloud_tgt,
		Eigen::Matrix4f& target2Src,
		bool downsample) {
	//
	// Downsample for consistency and speed
	// \note enable this for large datasets
	PointCloudType::Ptr src;
	PointCloudType::Ptr tgt;
	pcl::VoxelGrid<PointType> grid;
	if (downsample) {
		src = std::make_shared<PointCloudType>();
		grid.setLeafSize(0.05, 0.05, 0.05);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);

		tgt = std::make_shared<PointCloudType>();
		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	} else {
		src = cloud_src;
		tgt = cloud_tgt;
	}

	//
	// Align
	pcl::IterativeClosestPointNonLinear<PointType, PointType> reg;
	reg.setTransformationEpsilon (1e-6);

	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance (0.1);

	// Set the point representation
//	PointRep rep;
//	float rescale[3] = {1.0, 1.0, 1.0};
//	rep.setRescaleValues(rescale);
//	reg.setPointRepresentation(pcl::make_shared<const PointRep>(rep));

	// input data set
	reg.setInputSource(src);

	// cloud that we want to align the input source to
	reg.setInputTarget(tgt);

	reg.setMaximumIterations(60);
	reg.align(*src);

	target2Src = reg.getFinalTransformation().inverse();
}

PointCloudRegistration::PointCloudType::Ptr PointCloudRegistration::pair2Cloud(const ChannelReadPair& pair) {
	std::vector<cv::KeyPoint> keyPoints;
	m_fastDetector->detect(pair.first->getMat(), keyPoints);

	const int roiWidth = 3;
	const int roiHeight = 3;

	std::cout << "PCL Features: " << keyPoints.size() << std::endl;

	PointCloudType::Ptr newCloud = std::make_shared<PointCloudType>();
	newCloud->reserve(keyPoints.size() * roiWidth * roiHeight);

	const int rows = pair.second->getMat().rows;
	const int cols = pair.second->getMat().cols;
	const float* data = reinterpret_cast<const float*>(pair.second->data());

	for(const auto& point : keyPoints) {
		const auto& pt = point.pt;
		// get top left point
		int r = static_cast<int>(pt.y) - roiHeight/2;
		int c = static_cast<int>(pt.x) - roiWidth/2;

		if(r < 0 || r + roiHeight >= rows || c < 0 || c + roiWidth >= cols) {
			continue;
		}

		const float* depthVal = data + r * cols + c;

		for(int j = 0; j < roiHeight; ++j) {
			for(int i = 0; i < roiWidth; ++i) {
				float depth = *depthVal;
				PointType cloudPoint;
				if(getXYZ(r + j, c + i, depth, cloudPoint)) {
					newCloud->push_back(cloudPoint);
				}
				depthVal++;
			}
			depthVal += cols - roiWidth;
		}
	}

	return newCloud;
}

inline bool PointCloudRegistration::getXYZ(int r, int c, float depth, PointType& out) {
	if(!m_hasKinectRegData)
		return false;

	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(!std::isfinite(depth) || depth_val <= 0.001f) {
		//depth value is not valid
		return false;
	} else {
		out.x = (c + 0.5f - m_cx) * m_fx * depth_val;
		out.y = (r + 0.5f - m_cy) * m_fy * depth_val;
		out.z = depth_val;
		return true;
	}
}

//void PointCloudRegistration::onPoseChanged(ChannelRead<Pose2> data)
//{
	// TODO: this method is called whenever the pose has changed
//}

//void PointCloudRegistration::setPose(const Pose2& pose)
//{
	// TODO: this can be called by RPC (by other authorities, by user from RPC Console/View, mirainspect, ...)
//}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::PointCloudRegistration, mira::MicroUnit);
