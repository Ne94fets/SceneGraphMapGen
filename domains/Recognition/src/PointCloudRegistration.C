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
#include <pcl/common/distances.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//our visualizer
pcl::visualization::PCLVisualizer *p;
//its left and right viewports
int vp_1, vp_2;

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

	typedef cv::ORB			FeatureDetectorType;
	typedef cv::BFMatcher	FeatureMatcherType;

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
	void process();
	void processPair(const ChannelReadPair& pair);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

	bool pairAlignSrc2Target(const PointCloudType::Ptr cloud_src,
			const PointCloudType::Ptr cloud_tgt,
			Eigen::Matrix4f& src2target,
			bool downsample = false,
			const Eigen::Matrix4f* guess = nullptr);

	bool alignSrc2TargetRANSAC(const ChannelReadPair& pair,
							   Eigen::Matrix4f& src2target);

	PointCloudType::Ptr pair2Cloud(const ChannelReadPair& pair);
	PointCloudType::Ptr pair2CloudDepth(const ChannelReadPair& pair);
	bool getPoint(const cv::KeyPoint& kp, const DepthImgType& img, PointType& cloudPoint);
	bool getVaildPoint(int c, int r,
					   int roiWidth, int roiHeight,
					   const DepthImgType& img,
					   PointType& cloudPoint);
	bool getXYZ(int r, int c, float depth, PointType& out);

private:
	volatile bool m_shutdown = false;

	float	m_cx, m_cy;
	float	m_fx, m_fy;
	bool	m_hasKinectRegData = false;

	kinectdatatypes::RGBDQueue	m_rgbdQueue;

	cv::Ptr<FeatureDetectorType>	m_featureDetector;
	cv::Ptr<FeatureMatcherType>		m_featureMatcher;
	std::vector<cv::KeyPoint>		m_features;
	cv::Mat							m_descriptors;
	PointCloudType::Ptr				m_keyPoints3D;

	PointCloudType::Ptr				m_mapCloud;
	double							m_lastMapUpdateZRot;
	Eigen::Vector3f					m_lastMapUpdatePos;
	bool							m_lastConverged = false;
	int								m_maxIterations = 60;
	double							m_transformEps = 1e-8;
	double							m_euclidianFitnessEps = 1;
	double							m_maxCorrespndenceDist = 0.1;

	std::thread*	m_thread = nullptr;

	TransformType	m_globalTransform = TransformType::Identity();

	Channel<TransformType>	m_channelGlobalTransfrom;
	Channel<TransformType>	m_channelLocalTransform;
//	Channel<PointCloud<Eigen::Vector3f>	m_channelPointCloud;
};

///////////////////////////////////////////////////////////////////////////////

PointCloudRegistration::PointCloudRegistration() {
	// TODO: further initialization of members, etc.
	m_featureDetector = FeatureDetectorType::create();
	m_featureMatcher = FeatureMatcherType::create(cv::NormTypes::NORM_HAMMING, true);
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
	subscribe<DepthImgType>("DepthImageFull", &PointCloudRegistration::onDepthImage);

	// Create a PCLVisualizer object
	p = new pcl::visualization::PCLVisualizer ("Pairwise Incremental Registration");
	p->createViewPort (0.0, 0, 1.0, 1.0, vp_2);

	if(!m_thread) {
		m_thread = new std::thread([this](){ process(); });
	}
}

void PointCloudRegistration::onKinectRegistrationData(ChannelRead<KinectRegistrationData> data) {
	if(m_hasKinectRegData)
		return;

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
	if(!m_mapCloud) {
		m_mapCloud = pair2Cloud(pair);
		m_lastMapUpdatePos = Eigen::Vector3f(0,0,0);
		m_lastMapUpdateZRot = 0;

		Eigen::Vector4f currentPos = m_globalTransform * Eigen::Vector4f(0,0,0,1);
		Eigen::Vector4f view = m_globalTransform * Eigen::Vector4f(0,1,0,1);
		Eigen::Vector4f up = m_globalTransform * Eigen::Vector4f(0,0,1,1);
		p->setCameraPosition(currentPos.x(), currentPos.y(), currentPos.z(),
							 view.x(), view.y(), view.z(),
							 up.x(), up.y(), up.z(), vp_2);
		p->setCameraFieldOfView(90.f/180.f*M_PI, vp_2);

		PointCloudColorHandlerCustom<PointXYZ> cloud_tgt_h (m_mapCloud, 0, 255, 0);
		p->addPointCloud(m_mapCloud, cloud_tgt_h, "target", vp_2);

		PCL_INFO ("Press q to continue the registration.\n");
		p->spinOnce(10);
		return;
	}

	Eigen::Matrix4f source2targetTransform;
	PointCloudType::Ptr newCloud;

	auto startTime = std::chrono::system_clock::now();
	bool converged = true;
	if(alignSrc2TargetRANSAC(pair, source2targetTransform)) {
		newCloud = m_keyPoints3D;
	} else {
		// if fast is not possible try to recover
		newCloud = pair2Cloud(pair);
		converged = pairAlignSrc2Target(
					newCloud,
					m_mapCloud,
					source2targetTransform,
					false,
					m_lastConverged ? &m_globalTransform : nullptr);
		if(converged) {
			m_keyPoints3D = newCloud;
		}
	}
	auto endTime = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
	std::cout << "PCL align took: " << duration << "ms result: " << converged << std::endl;

	if(!converged) {
		m_lastConverged = false;
		return;
	}

	m_lastConverged = true;

	// get current time
	auto timestamp = Time::now();

	// publish local transfrom
	m_channelLocalTransform.post(Stamped<TransformType>(source2targetTransform, timestamp, pair.first->sequenceID));

	// accumulate transform
	m_globalTransform = source2targetTransform;
	m_channelGlobalTransfrom.post(Stamped<TransformType>(m_globalTransform, timestamp, pair.first->sequenceID));

	Eigen::Vector3f currentPos = m_globalTransform.topRightCorner(3, 1);
	Eigen::Vector3f xAxisDir = m_globalTransform.topLeftCorner(3, 1);
	double zRot = std::atan2(xAxisDir.y(), xAxisDir.x());
	std::cout << "Pcl rotation: " << zRot/M_PI*180 << std::endl;

	if((currentPos - m_lastMapUpdatePos).norm() > 1 || std::abs(zRot - m_lastMapUpdateZRot) > M_PI/20) {
		*m_mapCloud += *newCloud;
		m_lastMapUpdatePos = currentPos;
		m_lastMapUpdateZRot = zRot;
		std::cout << "Pcl update cloud" << std::endl;

		p->removePointCloud ("source");
		p->removePointCloud ("target");

		PointCloudColorHandlerCustom<PointXYZ> cloud_tgt_h (m_mapCloud, 0, 255, 0);
		PointCloudColorHandlerCustom<PointXYZ> cloud_src_h (newCloud, 255, 0, 0);
		p->addPointCloud(m_mapCloud, cloud_tgt_h, "target", vp_2);
		p->addPointCloud(newCloud, cloud_src_h, "source", vp_2);

		Eigen::Vector4f currentPos = m_globalTransform * Eigen::Vector4f(0,0,0,1);
		Eigen::Vector4f view = m_globalTransform * Eigen::Vector4f(0,1,0,1);
		Eigen::Vector4f up = m_globalTransform * Eigen::Vector4f(0,0,1,1);
		p->setCameraPosition(currentPos.x(), currentPos.y(), currentPos.z(),
							 view.x(), view.y(), view.z(),
							 up.x(), up.y(), up.z(), vp_2);

		PCL_INFO ("Press q to continue the registration.\n");
		p->spinOnce(10);
	}
}

bool PointCloudRegistration::pairAlignSrc2Target(
		const PointCloudType::Ptr cloud_src,
		const PointCloudType::Ptr cloud_tgt,
		Eigen::Matrix4f& src2target,
		bool downsample,
		const Eigen::Matrix4f* guess) {
	//
	// Downsample for consistency and speed
	// \note enable this for large datasets
	PointCloudType::Ptr src;
	PointCloudType::Ptr tgt;
	if (downsample) {
		pcl::VoxelGrid<PointType> grid;
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
	reg.setMaximumIterations(m_maxIterations);
	reg.setTransformationEpsilon (m_transformEps);
	reg.setEuclideanFitnessEpsilon(m_euclidianFitnessEps);

	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance (m_maxCorrespndenceDist);

	// input data set
	reg.setInputSource(src);

	// cloud that we want to align the input source to
	reg.setInputTarget(tgt);

	if(guess) {
		reg.align(*src, *guess);
	} else {
		reg.align(*src);
	}

	src2target = reg.getFinalTransformation();

	auto state = reg.getConvergeCriteria()->getConvergenceState();
	std::cout << "PCL Reg: FitScore: " << reg.getFitnessScore();
	switch (state) {
	case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_ITERATIONS:
		std::cout << "Reason: Iterations" << std::endl;
		break;
	case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_TRANSFORM:
		std::cout << "Reason: Transform" << std::endl;
		break;
	case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_ABS_MSE:
		std::cout << "Reason: ABS_MSE" << std::endl;
		break;
	case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_FAILURE_AFTER_MAX_ITERATIONS:
		std::cout << "Reason: Fail after max iterations" << std::endl;
		break;
	case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES:
		std::cout << "Reason: No correspondences" << std::endl;
		break;
	default:
		break;
	}

	return reg.hasConverged();
}

bool PointCloudRegistration::alignSrc2TargetRANSAC(const ChannelReadPair& pair,
												   Eigen::Matrix4f& src2target) {
	// cannot do RANSAC if there were no features in the previous run
	if(!m_keyPoints3D) {
		return false;
	}

	std::vector<cv::KeyPoint> features;
	m_featureDetector->detect(pair.first->getMat(), features);

	const int maxPoints = 300;
	cv::KeyPointsFilter::retainBest(features, maxPoints);

	if(features.size() < maxPoints) {
		std::cout << "PCL Features: " << features.size() << "/" << maxPoints << std::endl;
	}

	PointCloudType::Ptr keyPoints3D = std::make_shared<PointCloudType>();
	keyPoints3D->reserve(features.size());

	// inplace replace invalid points with valid points
	size_t validPointsCnt = 0;

	PointType cloudPoint;
	for(size_t i = 0; i < features.size(); ++i) {
		const cv::KeyPoint& kp = features[i];
		if(getPoint(kp, pair.second, cloudPoint)) {
			keyPoints3D->push_back(cloudPoint);
			features[validPointsCnt++] = kp;
		}
	}
	// erase copied valid and invalid points
	features.erase(features.begin() + validPointsCnt, features.end());

	cv::Mat descriptors;
	m_featureDetector->compute(pair.first->getMat(), features, descriptors);

	std::vector<cv::DMatch> matches;
	m_featureMatcher->match(descriptors, m_descriptors, matches);

	// random sample a few matches get with best euclidian distance match
	const size_t numSamples = 3;

	if(matches.size() < numSamples) {
		m_features.clear();
		m_keyPoints3D = nullptr;
		return false;
	}

	float minError = std::numeric_limits<float>::max();
	PointCloudType::Ptr tmpCloud = std::make_shared<PointCloudType>();
	for(size_t i = 0; i < 1000; ++i) {
		// permutate matches
		for(size_t j = 0; j < numSamples; ++j) {
			size_t idx = j + (std::rand() % (matches.size() - j));
			std::swap(matches[j], matches[idx]);
		}

		pcl::Correspondences corres;
		for(size_t j = 0; j < numSamples; ++j) {
			corres.push_back(pcl::Correspondence(matches[j].queryIdx, matches[j].trainIdx, matches[j].distance));
		}

		Eigen::Matrix4f tmpSrc2Tgt;
		pcl::registration::TransformationEstimationSVD<PointType, PointType> estimator;
		estimator.estimateRigidTransformation(*keyPoints3D, *m_keyPoints3D, corres, tmpSrc2Tgt);
		pcl::transformPointCloud(*keyPoints3D, *tmpCloud, tmpSrc2Tgt);

		float mse = 0;
		for(size_t j = 0; j < matches.size(); ++j) {
			const PointType& last = (*m_keyPoints3D)[matches[j].trainIdx];
			const PointType& cur = (*tmpCloud)[matches[j].queryIdx];
			mse += pcl::squaredEuclideanDistance(last, cur) / corres.size();
		}

		if(mse < minError) {
			minError = mse;
			src2target = tmpSrc2Tgt;
			std::cout << "PCL MSE: " << mse << std::endl;
		}
	}

	pcl::transformPointCloud(*keyPoints3D, *keyPoints3D, src2target);

	std::cout << "PCL MSE: " << minError << std::endl;
	if(minError > 5) {
		m_features.clear();
		m_keyPoints3D = nullptr;
		return false;
	}

	m_features = features;
	m_descriptors = descriptors;
	m_keyPoints3D = keyPoints3D;

	return true;
}

PointCloudRegistration::PointCloudType::Ptr PointCloudRegistration::pair2Cloud(const ChannelReadPair& pair) {
	std::vector<cv::KeyPoint> features;
	m_featureDetector->detect(pair.first->getMat(), features);

	const int maxPoints = 500;
	cv::KeyPointsFilter::retainBest(features, maxPoints);

	if(features.size() < maxPoints) {
		std::cout << "PCL Features: " << features.size() << "/" << maxPoints << std::endl;
	}

	PointCloudType::Ptr keyPoints3D = std::make_shared<PointCloudType>();
	keyPoints3D->reserve(features.size());

	// inplace replace invalid points with valid points
	size_t validPointsCnt = 0;

	PointType cloudPoint;
	for(size_t i = 0; i < features.size(); ++i) {
		const cv::KeyPoint& kp = features[i];
		if(getPoint(kp, pair.second, cloudPoint)) {
			keyPoints3D->push_back(cloudPoint);
			features[validPointsCnt++] = kp;
		}
	}
	// erase copied valid and invalid points
	features.erase(features.begin() + validPointsCnt, features.end());

	cv::Mat descriptors;
	m_featureDetector->compute(pair.first->getMat(), features, descriptors);

	m_features = features;
	m_descriptors = descriptors;

	return keyPoints3D;

//	const int roiWidth = 3;
//	const int roiHeight = 3;

//	PointCloudType::Ptr newCloud = std::make_shared<PointCloudType>();
//	newCloud->reserve(features.size() * roiWidth * roiHeight);

//	const int rows = pair.second->getMat().rows;
//	const int cols = pair.second->getMat().cols;

//	for(const auto& point : features) {
//		const auto& pt = point.pt;
//		// get top left point
//		int r = static_cast<int>(pt.y) - roiHeight/2;
//		int c = static_cast<int>(pt.x) - roiWidth/2;

//		if(r < 0 || r + roiHeight >= rows || c < 0 || c + roiWidth >= cols) {
//			continue;
//		}

//		PointType cloudPoint;
//		if(getVaildPoint(c, r, roiWidth, roiHeight, pair.second, cloudPoint)) {
//			newCloud->push_back(cloudPoint);
//		}
//	}

//	return newCloud;
}

PointCloudRegistration::PointCloudType::Ptr PointCloudRegistration::pair2CloudDepth(
		const ChannelReadPair& pair) {
	const int rows = pair.second->getMat().rows;
	const int cols = pair.second->getMat().cols;

	PointCloudType::Ptr newCloud = std::make_shared<PointCloudType>();
	newCloud->reserve((rows-4)/2 * (cols-600)/2);

	for(int r = 2; r < rows-2; r+=2) {
		const float* lineData = (*pair.second)[r];
		for(int c = 300; c < cols-300; c+=2) {
			const float *depthValue = lineData + c;
			PointType cloudPoint;
			if(getXYZ(r, c, *depthValue, cloudPoint)) {
				newCloud->push_back(cloudPoint);
			}
		}
	}
	return newCloud;
}

bool PointCloudRegistration::getPoint(const cv::KeyPoint& kp,
									  const DepthImgType& img,
									  PointType& cloudPoint) {
	const int r = static_cast<int>(kp.pt.y);
	const int c = static_cast<int>(kp.pt.x);

	assert(0 <= c && c < img.width() &&
		   0 <= r && r < img.height());

	const float depth = *(img[r] + c);

	return getXYZ(r, c, depth, cloudPoint);
}

bool PointCloudRegistration::getVaildPoint(int c, int r,
										   int roiWidth, int roiHeight,
										   const PointCloudRegistration::DepthImgType& img,
										   PointType& cloudPoint) {
	assert(0 <= c && c + roiWidth < img.width() &&
		   0 <= r && r + roiHeight < img.height());

	const float* depthVal = img[r] + c;

	for(int j = 0; j < roiHeight; ++j) {
		for(int i = 0; i < roiWidth; ++i) {
			float depth = *depthVal;
			if(getXYZ(r + j, c + i, depth, cloudPoint)) {
				return true;
			}
			depthVal++;
		}
		depthVal += img.width() - roiWidth;
	}

	return false;
}

inline bool PointCloudRegistration::getXYZ(int r, int c, float depth, PointType& out) {
	if(!m_hasKinectRegData)
		return false;

	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(!std::isfinite(depth) || depth_val <= 0.001f) {
		//depth value is not valid
		return false;
	}

	out.x = (c + 0.5f - m_cx) * m_fx * depth_val;
	out.z = -(r + 0.5f - m_cy) * m_fy * depth_val;
	out.y = depth_val;
	return true;
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
