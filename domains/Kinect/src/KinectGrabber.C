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
 * @file KinectGrabber.C
 *    Grabs RGB and depth images
 *
 * @author Steffen Kastner
 * @date   2019/10/01
 */

#include "KinectGrabber.h"

using namespace mira;

namespace kinect { 

KinectGrabber::KinectGrabber()
	: Unit(Duration::milliseconds(100)),
	  m_listener(libfreenect2::Frame::Color |
				 libfreenect2::Frame::Ir |
				 libfreenect2::Frame::Depth),
	  m_undistortedDepth(512, 424, 4),
	  m_registeredRGB(512, 424, 4),
	  m_bigdepth(1920, 1082, 4),
	  m_imgRGB(512, 424),
	  m_imgRGBFull(1920, 1080),
	  m_imgIR(512, 424),
	  m_imgDepth(512, 424),
	  m_imgDepthFull(1920, 1080)
{
	if(m_freenect2.enumerateDevices() == 0)
		throw std::runtime_error("No Kinect connected!");

	std::string serial = m_freenect2.getDefaultDeviceSerialNumber();
	m_dev = m_freenect2.openDevice(serial);

	m_dev->setColorFrameListener(&m_listener);
	m_dev->setIrAndDepthFrameListener(&m_listener);
}

KinectGrabber::~KinectGrabber()
{
	m_dev->stop();
	m_dev->close();

	if(m_registration)
		delete m_registration;

	delete m_dev;
}

void KinectGrabber::initialize()
{
	if(!m_dev->start())
		throw std::runtime_error("Could not start Kinect");

	std::cout << "Kinect serial: " << m_dev->getSerialNumber() << std::endl;
	std::cout << "Kinect firmware: " << m_dev->getFirmwareVersion() << std::endl;

	m_regData.depth_p = m_dev->getIrCameraParams();
	m_regData.rgb_p = m_dev->getColorCameraParams();

	m_registration = new libfreenect2::Registration(m_regData.depth_p,
													m_regData.rgb_p);

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &UnitName::onPoseChanged);
	m_channelRegistrationData = publish<RegistrationData>("KinectRegData");
	m_channelRGB = publish<RGBImgType>("RGBImage");
	m_channelRGBFull = publish<RGBImgType>("RGBImageFull");
	m_channelIR = publish<DepthImgType>("IRImage");
	m_channelDepth = publish<DepthImgType>("DepthImage");
	m_channelDepthFull = publish<DepthImgType>("DepthImageFull");
}

void KinectGrabber::process(const Timer& timer) {
	if(!m_listener.waitForNewFrame(m_frames, 10)) { //m_pollTime.totalMilliseconds()))
		std::cout << "No new frames in this time frame" << std::endl;
		return;
	}

	m_frameNumber++;

	auto wRegData = m_channelRegistrationData.write();
	wRegData->value() = m_regData;

	libfreenect2::Frame* rgb = m_frames[libfreenect2::Frame::Color];
	libfreenect2::Frame* ir = m_frames[libfreenect2::Frame::Ir];
	libfreenect2::Frame* depth = m_frames[libfreenect2::Frame::Depth];

	// post full rgb image
	cv::Mat rgbFull(rgb->height, rgb->width, CV_8UC4, rgb->data);
	cv::Mat tmpRGBFull(rgbFull.rows, rgbFull.cols, CV_8UC3);
	cv::cvtColor(rgbFull, tmpRGBFull, CV_BGRA2BGR);
	cv::flip(tmpRGBFull, m_imgRGBFull, 1);
	m_imgRGBFull.frameNumber() = m_frameNumber;

	ChannelWrite<RGBImgType> wRGBFull = m_channelRGBFull.write();
	wRGBFull->value() = m_imgRGBFull;

	// post ir image
	cv::Mat irMat(ir->height, ir->width, CV_32FC1, ir->data);
	cv::flip(irMat, m_imgIR, 1);
	m_imgIR.frameNumber() = m_frameNumber;

	ChannelWrite<DepthImgType> wIR = m_channelIR.write();
	wIR->value() = m_imgIR;

	// map rgb to depth and depth to rgb image
	m_registration->apply(rgb, depth,
						  &m_undistortedDepth, &m_registeredRGB, true,
						  &m_bigdepth);

	// post undistorted depth image (depth mapped to rgb)
	cv::Mat depthRaw(m_undistortedDepth.height, m_undistortedDepth.width,
					 CV_32FC1, m_undistortedDepth.data);
	cv::flip(depthRaw, m_imgDepth, 1);
	m_imgDepth.frameNumber() = m_frameNumber;

	ChannelWrite<DepthImgType> wDepth = m_channelDepth.write();
	wDepth->value() = m_imgDepth;

	// post full depth image (depth mapped to rgb)
	cv::Mat bigDepth(m_bigdepth.height, m_bigdepth.width, CV_32FC1, m_bigdepth.data);
	//bigDepth.copyTo(m_imgDepthFull);
	cv::flip(bigDepth, m_imgDepthFull, 1);
	m_imgDepthFull.frameNumber() = m_frameNumber;

	ChannelWrite<DepthImgType> wDepthFull = m_channelDepthFull.write();
	wDepthFull->value() = m_imgDepthFull;

	// post registered rgb image (rgb mapped to depth)
	cv::Mat bgrxReg(m_registeredRGB.height, m_registeredRGB.width,
					CV_8UC4, m_registeredRGB.data);
	cv::Mat tmpRGB(bgrxReg.rows, bgrxReg.cols, CV_8UC3);
	cv::cvtColor(bgrxReg, tmpRGB, CV_BGRA2BGR);
	cv::flip(tmpRGB, m_imgRGB, 1);
	m_imgRGB.frameNumber() = m_frameNumber;

	ChannelWrite<RGBImgType> wRGB = m_channelRGB.write();
	wRGB->value() = m_imgRGB;

	m_listener.release(m_frames);
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(kinect::KinectGrabber, mira::Unit );
