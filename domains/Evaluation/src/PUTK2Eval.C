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
 * @file PUTK2Eval.C
 *    Reads, simulates, evaluates PUTK2 data set
 *
 * @author Steffen Kastner
 * @date   2020/08/16
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/Unit.h>
#include <image/Img.h>

#include <kinectdatatypes/Types.h>

#include <algorithm>
#include <sstream>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace mira;

namespace evaluation { 

///////////////////////////////////////////////////////////////////////////////

/**
 * Reads, simulates, evaluates PUTK2 data set
 */
class PUTK2Eval : public Unit
{
MIRA_OBJECT(PUTK2Eval)
public:
	typedef kinectdatatypes::RegistrationData	RegistrationData;

	typedef Img<uint8_t, 3>	RGBImgType;
	typedef Img<float, 1>	DepthImgType;

	typedef Eigen::Matrix4f			TransformType;

public:

	PUTK2Eval();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, Unit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		r.property("Path to data set", m_path2DataSet, "Path to data set to evaluate", "external/evalData/Dataset_1_Kin_2/");
		r.property("Start", m_start, "Start evaluation", false);
		//r.member("Param2", mParam2, setter(&PUTK2Eval::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &PUTK2Eval::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

	virtual void process(const Timer& timer);

private:

	void onGlobalTransform(ChannelRead<TransformType> trans);

	void readPoses();

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:
	bool		m_start = false;
	bool		m_readPoses = false;

	size_t		m_imgSequenceID = 0;
	std::string	m_path2DataSet;

	RegistrationData	m_regData;
	TransformType		m_cam2RobotTransfrom = TransformType::Identity();

	double	m_posErr = 0;
	double	m_angleErr = 0;
	size_t	m_measurements = 0;

	Eigen::Matrix4d					m_initialPose = Eigen::Matrix4d::Identity();
	Eigen::Vector3d					m_initialPos;
	Eigen::Quaterniond				m_initialRot;
	std::vector<Eigen::Matrix4d>	m_poses;

	std::ofstream	m_outFile;

	Channel<RegistrationData>	m_channelRegistrationData;
	Channel<TransformType>		m_channelCam2RobotTransform;
	Channel<RGBImgType>			m_channelRGBFull;	// uchar range: [0,255]
	Channel<DepthImgType>		m_channelDepthFull;	// float unit: [mm] Non-positive, NaN, and infinity are invalid or missing data.
};

///////////////////////////////////////////////////////////////////////////////

PUTK2Eval::PUTK2Eval() : Unit(Duration::milliseconds(1000/10)) { // kinect has 30 FPS
//	m_regData.rgb_p.cx = 959.5;
//	m_regData.rgb_p.cy = 539.5;
//	m_regData.rgb_p.fx = 1081.3720703125;
//	m_regData.rgb_p.fy = 1081.3720703125;
	m_regData.rgb_p.cx = 952.6592286;
	m_regData.rgb_p.cy = 530.7386644;
	m_regData.rgb_p.fx = 1078.68499;
	m_regData.rgb_p.fy = 1076.4742562;

	m_cam2RobotTransfrom =
			Eigen::Affine3f(Eigen::AngleAxisf(1.29/180.0*M_PI, Eigen::Vector3f(0, 0, 1))).matrix() *
			Eigen::Affine3f(Eigen::AngleAxisf(14.00/180.0*M_PI, Eigen::Vector3f(1, 0, 0))).matrix() *
			Eigen::Affine3f(Eigen::AngleAxisf(-1.03/180.0*M_PI, Eigen::Vector3f(0, 1, 0))).matrix();

}

void PUTK2Eval::initialize() {
	m_channelRegistrationData = publish<RegistrationData>("KinectRegData");
	m_channelCam2RobotTransform = publish<TransformType>("Cam2RobotTransform");
	m_channelRGBFull = publish<RGBImgType>("RGBImageFull");
	m_channelDepthFull = publish<DepthImgType>("DepthImageFull");

	subscribe<TransformType>("PCGlobalTransform", &PUTK2Eval::onGlobalTransform);
}

void PUTK2Eval::process(const Timer& timer) {
	if(!m_start) {
		return;
	}
	if(!m_readPoses) {
		readPoses();
		m_readPoses = true;

		// open output file
		std::string path = m_path2DataSet + "out.txt";
		m_outFile.open(path);
		if(!m_outFile.is_open()) {
			throw std::runtime_error("Could not open: " + path);
		}
	}
	const auto captureTime = Time::now();

	auto wRegData = m_channelRegistrationData.write();
	wRegData->sequenceID = m_imgSequenceID;
	wRegData->timestamp = captureTime;
	wRegData->value() = m_regData;

	auto wCam2RobotTrans = m_channelCam2RobotTransform.write();
	wCam2RobotTrans->sequenceID = m_imgSequenceID;
	wCam2RobotTrans->timestamp = captureTime;
	wCam2RobotTrans->value() = m_cam2RobotTransfrom;

	std::stringstream sequenceID;
	sequenceID << std::setw(5) << std::setfill('0') << m_imgSequenceID;
	std::string rgbName = "rgb/rgb_" + sequenceID.str() + ".png";
	std::string depthName  = "depth/depth_" + sequenceID.str() + ".png";

	cv::Mat tmpRGB, tmpDepth;
	tmpRGB = cv::imread(m_path2DataSet + rgbName);
	tmpDepth = cv::imread(m_path2DataSet + depthName, cv::IMREAD_ANYDEPTH);

	if(tmpRGB.empty()) {
		m_start = false;
		m_imgSequenceID = 0;
		m_outFile.close();
		return;
	}

	RGBImgType rgb;
	DepthImgType depth;

	tmpRGB.copyTo(rgb);
	// convert 16-bit to float
	tmpDepth.convertTo(depth, CV_32F);

	ChannelWrite<RGBImgType> wRGBFull = m_channelRGBFull.write();
	wRGBFull->sequenceID = m_imgSequenceID;
	wRGBFull->timestamp = captureTime;
	wRGBFull->value() = rgb;

	ChannelWrite<DepthImgType> wDepthFull = m_channelDepthFull.write();
	wDepthFull->sequenceID = m_imgSequenceID;
	wDepthFull->timestamp = captureTime;
	wDepthFull->value() = depth;

	m_imgSequenceID++;
}

void PUTK2Eval::onGlobalTransform(ChannelRead<TransformType> trans) {
	Eigen::Matrix4d robotTrans = trans->cast<double>();

	Eigen::Vector4d globalPos = robotTrans.topRightCorner(4, 1);

	Eigen::Matrix3d onlyRot = robotTrans.topLeftCorner(3,3);
	Eigen::Quaterniond globalRot(onlyRot);

	m_outFile << trans->sequenceID << " " <<
				 globalPos.x() << " " <<
				 globalPos.y() << " " <<
				 globalPos.z() << " " <<
				 globalRot.x() << " " <<
				 globalRot.y() << " " <<
				 globalRot.z() << " " <<
				 globalRot.w() << std::endl;
}

void PUTK2Eval::readPoses() {
	std::string path = m_path2DataSet + "groundtruth.txt";
	std::ifstream file(path);
	if(!file.is_open()) {
		throw std::runtime_error("Could not open file: " + path);
	}

	bool initialPosSet = false;

	std::string line;
	while(std::getline(file, line)) {
		size_t id;
		Eigen::Vector3d pos;
		Eigen::Quaterniond q;
		std::stringstream ss(line);
		ss >> id >> pos.x() >> pos.y() >> pos.z() >> q.x() >> q.y() >> q.z() >> q.w();

		Eigen::Matrix4d t =
				Eigen::Affine3d(Eigen::Translation3d(pos)).matrix() *
				Eigen::Affine3d(q).matrix();

		m_poses.push_back(t);

		if(!initialPosSet) {
			std::cout << t << std::endl;
			m_initialPose = t;
			m_initialPos = pos;
			m_initialRot = q;
			initialPosSet = true;
		}
	}
}

//void PUTK2Eval::onPoseChanged(ChannelRead<Pose2> data)
//{
	// TODO: this method is called whenever the pose has changed
//}

//void PUTK2Eval::setPose(const Pose2& pose)
//{
	// TODO: this can be called by RPC (by other authorities, by user from RPC Console/View, mirainspect, ...)
//}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(evaluation::PUTK2Eval, mira::Unit);
