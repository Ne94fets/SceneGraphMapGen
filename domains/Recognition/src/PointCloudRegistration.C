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
#include <future>
#include <chrono>

#include <pcl/point_cloud.h>

#include <pcl/memory.h>
#include <pcl/common/distances.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_2D.h>

#include <pcl/visualization/pcl_visualizer.h>
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <kinectdatatypes/Types.h>
#include <kinectdatatypes/RGBDQueue.h>
#include <kinectdatatypes/RGBDOperations.h>
using kinectdatatypes::RGBDOperations;

#define DEBUG_MATCHES 0
#define DEBUG_POINT_CLOUD 0

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
	typedef PointCloud<PointXYZRGB>	RGBPointCloudType;

	typedef kinectdatatypes::RGBDQueue<Stamped<RGBImgType>, Stamped<DepthImgType>>	SyncQueueType;
	typedef typename SyncQueueType::ChannelPair										ChannelPair;

	typedef cv::ORB			FeatureDetectorType;
	typedef cv::BFMatcher	FeatureMatcherType;

	struct HistoryEntry {
		std::vector<cv::KeyPoint>	features;
		cv::Mat						descriptors;
		PointCloudType::Ptr			keyPoints3D;
		cv::Mat						img;
		size_t						age;
	};

public:

	PointCloudRegistration();
	~PointCloudRegistration();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		r.property("MaxIterations", m_maxIterations, "Maximum pcl align iterations", 60);
//		r.property("TransformEps", m_transformEps, "Max allowable translation squared between two consecutive transforms", 1e-6);
//		r.property("EuclideanFitnessEps", m_euclidianFitnessEps, "Max allowable eucledian transform between two consecutive transforms", 1e-6);
		r.property("MaxCorrespondenceDist", m_maxCorrespndenceDist, "Max distance between two correspondences", 0.1);
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
	void onCam2RobotTransform(ChannelRead<TransformType> trans);

	void visualize();
	void process();
	void processPair(const ChannelPair& pair);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

	bool getTransformRANSAC(const HistoryEntry& newEntry,
							const HistoryEntry& entry,
							const ChannelPair& pair,
							Eigen::Matrix4f& src2target,
							size_t& inliers);

	void pair2Cloud(const ChannelPair& pair, HistoryEntry& newEntry);
	PointCloudType::Ptr pair2CloudDepth(const ChannelPair& pair);
	RGBPointCloudType::Ptr pair2CloudDepthColor(const ChannelPair& pair);
	void getCloud(std::vector<cv::KeyPoint>& features,
				  cv::Mat& descriptors,
				  const PointCloudType::Ptr cloud,
				  const DepthImgType& img);
	bool getPoint(const cv::KeyPoint& kp, const DepthImgType& img, PointType& cloudPoint);
	bool getNearestPointIn(int c, int r,
						   int roiWidth, int roiHeight,
						   const DepthImgType& img,
						   PointType& cloudPoint);
	bool getVaildPointIn(int c, int r,
						 int roiWidth, int roiHeight,
						 const DepthImgType& img,
						 PointType& cloudPoint);
	bool getXYZ(float r, float c, float depth, PointType& out);

private:
	volatile bool m_shutdown = false;

	float	m_cx, m_cy;
	float	m_fx, m_fy;
	bool	m_hasKinectRegData = false;

	SyncQueueType	m_rgbdQueue;

	size_t							m_featureMaxPoints = 50;
	cv::Ptr<FeatureDetectorType>	m_featureDetector;
	cv::Ptr<FeatureMatcherType>		m_featureMatcher;

	size_t							m_historyStatic = 3;
	std::vector<HistoryEntry>		m_history;

	std::mutex					m_mapMutex;
	volatile bool				m_update = false;
	RGBPointCloudType::Ptr		m_fullCloud;
	RGBPointCloudType::Ptr		m_newFullCloud;
	PointCloudType::Ptr			m_mapCloud;
	PointCloudType::Ptr			m_newCloud;

	double							m_lastMapUpdateZRot;
	Eigen::Vector3f					m_lastMapUpdatePos;
	bool							m_lastConverged = false;
	int								m_maxIterations = 60;
	double							m_transformEps = 1e-8;
	double							m_euclidianFitnessEps = 1;
	double							m_maxCorrespndenceDist = 0.1;

	std::thread*	m_thread = nullptr;
	std::thread*	m_visThread = nullptr;

	TransformType	m_cam2RobotTransform = TransformType::Identity();

	TransformType	m_globalTransform = TransformType::Identity();
	TransformType	m_localTransform = TransformType::Identity();

	Channel<TransformType>	m_channelGlobalTransfrom;
	Channel<TransformType>	m_channelLocalTransform;
//	Channel<PointCloud<Eigen::Vector3f>	m_channelPointCloud;

	size_t	m_measurementCnt = 0;
	size_t	m_accumulatedProcessingTime = 0;
};

///////////////////////////////////////////////////////////////////////////////

PointCloudRegistration::PointCloudRegistration() {
	// TODO: further initialization of members, etc.
	m_featureDetector = FeatureDetectorType::create(m_featureMaxPoints);
	m_featureMatcher = FeatureMatcherType::create(cv::NormTypes::NORM_HAMMING, true);
	m_history.resize(m_historyStatic);

	m_fullCloud = std::make_shared<RGBPointCloudType>();
	m_newFullCloud = std::make_shared<RGBPointCloudType>();
	m_mapCloud = std::make_shared<PointCloudType>();
	m_newCloud = std::make_shared<PointCloudType>();
}

PointCloudRegistration::~PointCloudRegistration() {
	m_shutdown = true;

	if(m_visThread) {
		m_visThread->join();
		delete m_visThread;
	}

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
	subscribe<TransformType>("Cam2RobotTransform", &PointCloudRegistration::onCam2RobotTransform);
	subscribe<RGBImgType>("RGBImageFull", &PointCloudRegistration::onRGBImage);
	subscribe<DepthImgType>("DepthImageFull", &PointCloudRegistration::onDepthImage);

	if(!m_thread) {
		m_thread = new std::thread([this](){ process(); });
	}

#if DEBUG_POINT_CLOUD
	if(!m_visThread) {
		m_visThread = new std::thread([this]() { visualize(); });
	}
#endif
}

void PointCloudRegistration::onKinectRegistrationData(ChannelRead<KinectRegistrationData> data) {
	if(m_hasKinectRegData)
		return;

	m_cx = data->rgb_p.cx;
	m_cy = data->rgb_p.cy;
	m_fx = 1 / data->rgb_p.fx;
	m_fy = 1 / data->rgb_p.fy;
//	m_cx = data->depth_p.cx;
//	m_cy = data->depth_p.cy;
//	m_fx = 1 / data->depth_p.fx;
//	m_fy = 1 / data->depth_p.fy;
	m_hasKinectRegData = true;
}

void PointCloudRegistration::onDepthImage(ChannelRead<DepthImgType> image) {
	static size_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing depth image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	Stamped<DepthImgType> stamped(image);
	m_rgbdQueue.push1(stamped);
}

void PointCloudRegistration::onRGBImage(ChannelRead<RGBImgType> image) {
	static uint32_t prevNumber = 0;
	if(image->sequenceID != prevNumber+1) {
		std::cout << "Missing color image " << prevNumber+1 << " til " << image->sequenceID << std::endl;
	}
	prevNumber = image->sequenceID;
	Stamped<RGBImgType> stamped(image);
	m_rgbdQueue.push0(stamped);
}

void PointCloudRegistration::onCam2RobotTransform(ChannelRead<PointCloudRegistration::TransformType> trans) {
	m_cam2RobotTransform = trans;
}

void PointCloudRegistration::visualize() {
	int	vpKeypoint, vpFull;
	pcl::visualization::PCLVisualizer::Ptr visualizer = std::make_shared<pcl::visualization::PCLVisualizer>("Registration");
	visualizer->createViewPort(0.0, 0.0, 1.0, 0.5, vpKeypoint);
	visualizer->createViewPort(0.0, 0.5, 1.0, 1.0, vpFull);

	{
		std::lock_guard guard(m_mapMutex);

		PointCloudColorHandlerCustom<PointXYZ> cloud_tgt_h (m_mapCloud, 0, 255, 0);
		PointCloudColorHandlerCustom<PointXYZ> cloud_src_h (m_newCloud, 255, 0, 0);
		visualizer->addPointCloud(m_mapCloud, cloud_tgt_h, "target", vpKeypoint);
		visualizer->addPointCloud(m_newCloud, cloud_src_h, "source", vpKeypoint);
		visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_tgt_full_h (m_fullCloud);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_src_full_h (m_newFullCloud);
		visualizer->addPointCloud(m_fullCloud, cloud_tgt_full_h, "fullTarget", vpFull);
		visualizer->addPointCloud(m_newFullCloud, cloud_src_full_h, "fullSource", vpFull);
		visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "fullTarget");
		visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "fullSource");
	}

	while(!visualizer->wasStopped() && !m_shutdown) {
		if(m_update) {
			std::lock_guard guard(m_mapMutex);
			m_update = false;

			PointCloudColorHandlerCustom<PointXYZ> cloud_tgt_h (m_mapCloud, 0, 255, 0);
			PointCloudColorHandlerCustom<PointXYZ> cloud_src_h (m_newCloud, 255, 0, 0);
			visualizer->updatePointCloud(m_mapCloud, cloud_tgt_h, "target");
			visualizer->updatePointCloud(m_newCloud, cloud_src_h, "source");

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_tgt_full_h (m_fullCloud);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_src_full_h (m_newFullCloud);
			visualizer->updatePointCloud(m_fullCloud, cloud_tgt_full_h, "fullTarget");
			visualizer->updatePointCloud(m_newFullCloud, cloud_src_full_h, "fullSource");

			Eigen::Vector4f currentPos = m_globalTransform * Eigen::Vector4f(0,0,0,1);
			Eigen::Vector4f view = m_globalTransform * Eigen::Vector4f(0,1,0,1);
			Eigen::Vector4f up = m_globalTransform * Eigen::Vector4f(0,0,1,0);
			Eigen::Vector4f back = m_globalTransform * Eigen::Vector4f(0,0,0,0);
			visualizer->setCameraPosition(currentPos.x() + back.x(), currentPos.y() + back.y(), currentPos.z() + back.z(),
											view.x(), view.y(), view.z(),
											up.x(), up.y(), up.z());
			visualizer->setCameraFieldOfView(60.f/180.f*M_PI, vpKeypoint);
		}
		visualizer->spinOnce(100);
	}
}

void PointCloudRegistration::process() {
	while(!m_shutdown && !m_hasKinectRegData) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	std::cout << "PointCloudRegistration has registration data. Running in main loop now." << std::endl;

	while(!m_shutdown) {

		// wait for a matching pair
		const auto pair = m_rgbdQueue.getNewestSyncedPair();

		auto startTime = std::chrono::system_clock::now();

		// process the pair
		processPair(pair);

		auto endTime = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		m_measurementCnt++;
		m_accumulatedProcessingTime += duration;
		std::cout << "PCL avg process time: " << (double)m_accumulatedProcessingTime / m_measurementCnt << std::endl;
		if(duration > 1000/30)
			std::cout << "PCL process took: " << duration << "ms" << std::endl;
	}
}

void PointCloudRegistration::processPair(const ChannelPair& pair) {
	// init in first run
	if(!m_history[0].keyPoints3D) {
		HistoryEntry newEntry;
		pair2Cloud(pair, newEntry);
		m_history[0] = newEntry;
		m_history.insert(m_history.begin() + m_historyStatic, newEntry);

#if DEBUG_POINT_CLOUD
		{
			std::lock_guard guard(m_mapMutex);

			m_mapCloud->insert(m_mapCloud->begin(), newEntry.keyPoints3D->begin(), newEntry.keyPoints3D->end());

			m_fullCloud = pair2CloudDepthColor(pair);
			m_update = true;
		}
#endif

		m_lastMapUpdatePos = Eigen::Vector3f(0,0,0);
		m_lastMapUpdateZRot = 0;
		return;
	}

	// align clouds
//	auto startTime = std::chrono::system_clock::now();
	Eigen::Matrix4f source2targetTransform;
	HistoryEntry newEntry;
	pair2Cloud(pair, newEntry);
//	auto endTime = std::chrono::system_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
//	std::cout << "PCL pair2Cloud took: " << duration << "ms" << std::endl;

//	startTime = std::chrono::system_clock::now();
	bool converged = false;
	size_t maxInliers = 0;
	for(size_t i = 0; i < m_historyStatic && !m_history[i].features.empty(); ++i) {
		size_t inliers;
		TransformType tmpSrc2Target;
		bool tmpConverged = getTransformRANSAC(newEntry, m_history[i], pair, tmpSrc2Target, inliers);
		if(tmpConverged && inliers > maxInliers) {
			converged = tmpConverged;
			maxInliers = inliers;
			source2targetTransform = tmpSrc2Target;
		}
	}

	// search in global history if not found in recent images
	if(!converged) {
		for(size_t i = m_historyStatic; i < m_history.size() && !m_history[i].features.empty(); ++i) {
			size_t inliers;
			TransformType tmpSrc2Target;
			bool tmpConverged = getTransformRANSAC(newEntry, m_history[i], pair, tmpSrc2Target, inliers);
			if(tmpConverged && inliers > maxInliers) {
				converged = tmpConverged;
				maxInliers = inliers;
				source2targetTransform = tmpSrc2Target;
			}
		}
	}

//	endTime = std::chrono::system_clock::now();
//	duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
//	std::cout << "PCL align took: " << duration << "ms result: " << converged << std::endl;

	// if not aligned return
	if(!converged) {
		m_lastConverged = false;
		return;
	}

	// if aligned let history age and put into history
	m_lastConverged = true;
	pcl::transformPointCloud(*newEntry.keyPoints3D, *newEntry.keyPoints3D, source2targetTransform);

	for(size_t i = m_historyStatic-1; i > 0; --i) {
		m_history[i] = m_history[i-1];
	}
	m_history[0] = newEntry;

	// get current time
	auto timestamp = Time::now();

	// publish local transfrom
	m_localTransform = m_globalTransform.transpose() * source2targetTransform;
	m_channelLocalTransform.post(Stamped<TransformType>(m_localTransform, timestamp, pair.first.sequenceID));

	// publish global transform
	m_globalTransform = source2targetTransform;
	m_channelGlobalTransfrom.post(Stamped<TransformType>(m_globalTransform, timestamp, pair.first.sequenceID));

	Eigen::Vector3f currentPos = m_globalTransform.topRightCorner(3, 1);
	Eigen::Vector3f xAxisDir = m_globalTransform.topLeftCorner(3, 1);
	double zRot = std::atan2(xAxisDir.y(), xAxisDir.x());

	// update global history after driving 0.5m or after rotating 9 degrees
	const float updateDist = 0.5;
	const float updateRot = M_PI/32;
	if((currentPos - m_lastMapUpdatePos).norm() > updateDist ||
			std::abs(zRot - m_lastMapUpdateZRot) > updateRot) {

		m_history.insert(m_history.begin() + m_historyStatic, newEntry);
		m_lastMapUpdatePos = currentPos;
		m_lastMapUpdateZRot = zRot;

#if DEBUG_POINT_CLOUD
		{
			std::lock_guard guard(m_mapMutex);

			RGBPointCloudType::Ptr newFullCloud = pair2CloudDepthColor(pair);
			pcl::transformPointCloud(*newFullCloud, *newFullCloud, m_globalTransform);
			m_newFullCloud = newFullCloud;
			*m_fullCloud += *newFullCloud;

			m_newCloud = newEntry.keyPoints3D;
			*m_mapCloud += *newEntry.keyPoints3D;

			m_update = true;
		}
#endif

		std::cout << "PCL update cloud" << std::endl;
	}
}

void transform(PointCloudRegistration::PointType& p,
			   const PointCloudRegistration::TransformType& t) {
	Eigen::Vector4f vec(p.x, p.y, p.z, 1);
	vec = t * vec;
	p.x = vec.x();
	p.y = vec.y();
	p.z = vec.z();
}

bool PointCloudRegistration::getTransformRANSAC(const HistoryEntry& newEntry,
												const HistoryEntry& entry,
												const ChannelPair& pair,
												Eigen::Matrix4f& src2target,
												size_t& inliers) {
	// cannot do RANSAC if there were no features in the previous run
	if(!entry.keyPoints3D) {
		return false;
	}

	std::vector<cv::DMatch> matches;
	m_featureMatcher->match(newEntry.descriptors, entry.descriptors, matches);

	int pixelPerMatch = 50;
	const int bitMaskWidth = pair.first.width() / pixelPerMatch + 1;
	const int bitMaskHeight = pair.first.height() / pixelPerMatch + 1;
	std::vector<bool> bitMask(bitMaskWidth * bitMaskHeight, false);

	size_t validCnt = 0;
	for(size_t i = 0; i < matches.size(); ++i) {
		const auto& p = newEntry.features[matches[i].queryIdx];
		const int maskPos =
				static_cast<int>(p.pt.y) / pixelPerMatch * bitMaskWidth +
				static_cast<int>(p.pt.x) / pixelPerMatch;
		if(!bitMask[maskPos]) {
			matches[validCnt] = matches[i];
			validCnt++;
			bitMask.at(maskPos) = true;
		}
	}

	matches.erase(matches.begin() + validCnt, matches.end());

#if DEBUG_MATCHES
	cv::Mat outImg;
	cv::drawMatches(pair.first.clone(), newEntry.features, entry.img, entry.features, matches, outImg);

	cv::imshow("Matches", outImg);
#endif

	// random sample a few matches to get some with best euclidian distance match
	const size_t numSamples = 3;

	if(matches.size() < 10) {
		return false;
	}

	pcl::registration::TransformationEstimation2D<PointType, PointType> estimator;

	size_t maxInliers = 0;
	pcl::Correspondences inlCorres;
	// RANSAC draws = log(1 - z) / log(1 - w^n)
	// z = 0.99, w = 0.5 (50% of matches are good), n = numSamples
	const double probErrorFreeSample = 0.99;
	const double probMatchGood = 0.20;
	const size_t maxDraws = static_cast<int>(std::ceil(std::log(1 - probErrorFreeSample) /
													   std::log(1 - std::pow(probMatchGood, numSamples))));
	for(size_t i = 0; i < maxDraws; ++i) {
		// permutate matches
		for(size_t j = 0; j < numSamples; ++j) {
			size_t idx = j + (std::rand() % (matches.size() - j));
			std::swap(matches[j], matches[idx]);
		}

		bool pass = true;
		for(size_t j = 0; j < numSamples; ++j) {
			const PointType& last0 = (*entry.keyPoints3D)[matches[j].trainIdx];
			const PointType& last1 = (*entry.keyPoints3D)[matches[(j+1)%numSamples].trainIdx];
			const float lastDist = pcl::euclideanDistance(last0, last1);
			const PointType& cur0 = (*newEntry.keyPoints3D)[matches[j].queryIdx];
			const PointType& cur1 = (*newEntry.keyPoints3D)[matches[(j+1)%numSamples].queryIdx];
			const float curDist = pcl::euclideanDistance(cur0, cur1);
			float error = std::abs(curDist - lastDist);
			float errorPercentage = error / ((curDist + lastDist) / 2);
			if(errorPercentage > 0.1) {
				pass = false;
				break;
			}
		}

		if(!pass) {
			continue;
		}

		pcl::Correspondences corres;
		for(size_t j = 0; j < numSamples; ++j) {
			corres.push_back(pcl::Correspondence(matches[j].queryIdx, matches[j].trainIdx, matches[j].distance));
		}

		Eigen::Matrix4f tmpSrc2Tgt;
		estimator.estimateRigidTransformation(*newEntry.keyPoints3D, *entry.keyPoints3D, corres, tmpSrc2Tgt);


		size_t inliersCnt = 0;
		for(size_t j = 0; j < matches.size(); ++j) {
			const PointType& last = (*entry.keyPoints3D)[matches[j].trainIdx];
			PointType cur = (*newEntry.keyPoints3D)[matches[j].queryIdx];
			transform(cur, tmpSrc2Tgt);
			float dist = pcl::euclideanDistance(last, cur);
			if(dist < 0.03) {
				inliersCnt++;
				corres.push_back(pcl::Correspondence(matches[j].queryIdx, matches[j].trainIdx, matches[j].distance));
			}
		}

		if(inliersCnt > maxInliers) {
			maxInliers = inliersCnt;
			src2target = tmpSrc2Tgt;
			inlCorres = std::move(corres);
		}
	}

	if(maxInliers < matches.size() * probMatchGood) {
		return false;
	}

	inliers = maxInliers;

	estimator.estimateRigidTransformation(*newEntry.keyPoints3D, *entry.keyPoints3D, inlCorres, src2target);

	return true;
}

void PointCloudRegistration::pair2Cloud(const ChannelPair& pair,
										HistoryEntry& newEntry) {
	typedef std::tuple<std::vector<cv::KeyPoint>, cv::Mat, int, int> FutureReturnType;
	std::vector<std::future<FutureReturnType>> futures;

	// detectAndCompute is faster than detecting, discarding at getCloud and compute afterwards
	int stepHeight = pair.first.height()/2;
	int stepWidth = pair.first.width()/2;
	for(int y = 0; y < pair.first.height(); y += stepHeight) {
		for(int x = 0; x < pair.first.width(); x += stepWidth) {
			cv::Mat part = pair.first.getMat()(cv::Rect(x, y, stepWidth, stepHeight));
			futures.push_back(std::async([this](const cv::Mat& part, int x, int y){
				std::vector<cv::KeyPoint> features;
				cv::Mat descriptors;
				m_featureDetector->detectAndCompute(part, cv::Mat(), features, descriptors);
				return std::make_tuple(features, descriptors, x, y);
			}, part, x, y));
		}
	}

	std::vector<cv::KeyPoint> features;
	cv::Mat descriptors;

	for(auto& f : futures) {
		FutureReturnType out = f.get();
		std::vector<cv::KeyPoint>& tmpFeatures = std::get<0>(out);
		cv::Mat& tmpDescriptors = std::get<1>(out);
		int x = std::get<2>(out), y = std::get<3>(out);
		features.reserve(features.size() + tmpFeatures.size());
		for(auto& feat : tmpFeatures) {
			feat.pt.x += x;
			feat.pt.y += y;
			features.push_back(feat);
		}
		descriptors.push_back(tmpDescriptors);
	}

	if(features.size() < m_featureMaxPoints) {
		std::cout << "PCL Features: " << features.size() << "/" << m_featureMaxPoints << std::endl;
	}

	PointCloudType::Ptr keyPoints3D = std::make_shared<PointCloudType>();
	getCloud(features, descriptors, keyPoints3D, pair.second);

	newEntry.age = 0;

#if DEBUG_MATCHES
	newEntry.img = pair.first.clone();
#endif

	newEntry.features = features;
	newEntry.descriptors = descriptors;
	newEntry.keyPoints3D = keyPoints3D;
}

PointCloudRegistration::PointCloudType::Ptr PointCloudRegistration::pair2CloudDepth(
		const ChannelPair& pair) {
	const int rows = pair.second.getMat().rows;
	const int cols = pair.second.getMat().cols;

	PointCloudType::Ptr newCloud = std::make_shared<PointCloudType>();
	newCloud->reserve((rows-4)/2 * (cols-600)/2);

	for(int r = 2; r < rows-2; r+=2) {
		const float* lineData = pair.second[r];
		for(int c = 300; c < cols-300; c+=2) {
			const float *depthValue = lineData + c;
			PointType cloudPoint;
			if(RGBDOperations::getXYZ(r, c, *depthValue, m_cx, m_cy, m_fx, m_fy, cloudPoint)) {
				transform(cloudPoint, m_cam2RobotTransform);
				newCloud->push_back(cloudPoint);
			}
		}
	}
	return newCloud;
}

PointCloudRegistration::RGBPointCloudType::Ptr PointCloudRegistration::pair2CloudDepthColor(
		const ChannelPair& pair) {
	const int rows = pair.second.getMat().rows;
	const int cols = pair.second.getMat().cols;

	RGBPointCloudType::Ptr newCloud = std::make_shared<RGBPointCloudType>();
	newCloud->reserve((rows-4)/2 * (cols-600)/2);

	for(int r = 2; r < rows-2; r+=2) {
		const float* lineData = pair.second[r];
		for(int c = 300; c < cols-300; c+=2) {
			const float *depthValue = lineData + c;
			PointXYZ cloudPoint;
			if(RGBDOperations::getXYZ(r, c, *depthValue, m_cx, m_cy, m_fx, m_fy, cloudPoint)) {
				RGBImgType::Pixel pix = pair.first(c, r);
				PointXYZRGB colorPoint(cloudPoint.x,
									   cloudPoint.y,
									   cloudPoint.z,
									   pix[2],
									   pix[1],
									   pix[0]);

				newCloud->push_back(colorPoint);
			}
		}
	}

	pcl::transformPointCloud(*newCloud, *newCloud, m_cam2RobotTransform);

	return newCloud;
}

void PointCloudRegistration::getCloud(std::vector<cv::KeyPoint>& features,
									  cv::Mat& descriptors,
									  const PointCloudType::Ptr cloud,
									  const DepthImgType& img) {
	cloud->reserve(features.size());

	// inplace replace invalid points with valid points
	size_t validPointsCnt = 0;

	PointType cloudPoint;
	for(size_t i = 0; i < features.size(); ++i) {
		const cv::KeyPoint& kp = features[i];
		if(getNearestPointIn(kp.pt.x, kp.pt.y, 3, 3, img, cloudPoint)) {
			// move point relative to robot
			transform(cloudPoint, m_cam2RobotTransform);

			// add to cloud
			cloud->push_back(cloudPoint);

			// save features to point
			features[validPointsCnt] = kp;
			descriptors.row(i).copyTo(descriptors.row(validPointsCnt));
			validPointsCnt++;
		}
	}
	// erase copied valid and invalid points
	features.resize(validPointsCnt);
	descriptors.resize(validPointsCnt);
}

bool PointCloudRegistration::getPoint(const cv::KeyPoint& kp,
									  const DepthImgType& img,
									  PointType& cloudPoint) {
	const float x = kp.pt.x;
	const float y = kp.pt.y;
	const int r = static_cast<int>(y);
	const int c = static_cast<int>(x);

	assert(0 <= c && c + 1< img.width() &&
		   0 <= r && r + 1 < img.height());

	const float depth00 = *(img[r] + c);
	const float depth01 = *(img[r] + c + 1);
	const float depth10 = *(img[r + 1] + c);
	const float depth11 = *(img[r + 1] + c + 1);
	const float cRatio = x-c;
	const float depth0 = depth00 * (1-cRatio) + depth01 * cRatio;
	const float depth1 = depth10 * (1-cRatio) + depth11 * cRatio;
	const float rRatio = y-r;
	const float depth = depth0 * (1-rRatio) - depth1 * rRatio;

	return RGBDOperations::getXYZ(r, c, depth, m_cx, m_cy, m_fx, m_fy, cloudPoint);
}

bool PointCloudRegistration::getNearestPointIn(int c, int r,
											   int roiWidth, int roiHeight,
											   const DepthImgType& img,
											   PointType& cloudPoint) {
	assert(0 <= c && c + roiWidth < img.width() &&
		   0 <= r && r + roiHeight < img.height());

	const float* depthVal = img[r] + c;

	float nearest = std::numeric_limits<float>::infinity();
	int row, col;
	for(int j = 0; j < roiHeight; ++j) {
		for(int i = 0; i < roiWidth; ++i) {
			float depth = *depthVal;
			if(depth < nearest) {
				nearest = depth;
				row = r + j;
				col = c + i;
			}
			depthVal++;
		}
		depthVal += img.width() - roiWidth;
	}

	return RGBDOperations::getXYZ(row, col, nearest, m_cx, m_cy, m_fx, m_fy, cloudPoint);
}

bool PointCloudRegistration::getVaildPointIn(int c, int r,
										   int roiWidth, int roiHeight,
										   const PointCloudRegistration::DepthImgType& img,
										   PointType& cloudPoint) {
	assert(0 <= c && c + roiWidth < img.width() &&
		   0 <= r && r + roiHeight < img.height());

	const float* depthVal = img[r] + c;

	for(int j = 0; j < roiHeight; ++j) {
		for(int i = 0; i < roiWidth; ++i) {
			float depth = *depthVal;
			if(RGBDOperations::getXYZ(r + j, c + i, depth, m_cx, m_cy, m_fx, m_fy, cloudPoint)) {
				return true;
			}
			depthVal++;
		}
		depthVal += img.width() - roiWidth;
	}

	return false;
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
