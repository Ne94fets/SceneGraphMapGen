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

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <fw/Unit.h>
#include <image/Img.h>

#include <exception>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace mira;

namespace kinect { 

///////////////////////////////////////////////////////////////////////////////

/**
 * Grabs RGB and depth images
 */
class KinectGrabber : public Unit
{
MIRA_OBJECT(KinectGrabber)

public:

	KinectGrabber();
	virtual ~KinectGrabber();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, Unit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&UnitName::setParam2,this), "Second parameter with setter");
	}

protected:

	virtual void initialize();

	virtual void process(const Timer& timer);

private:

	// void onPoseChanged(ChannelRead<Pose2> pose);

private:
	libfreenect2::Freenect2					m_freenect2;
	libfreenect2::Freenect2Device*			m_dev = nullptr;
	libfreenect2::Registration*				m_registration = nullptr;
	libfreenect2::SyncMultiFrameListener	m_listener;
	libfreenect2::FrameMap					m_frames;
	libfreenect2::Frame						m_undistortedDepth;
	libfreenect2::Frame						m_registeredRGB;

	typedef Img<uint8_t, 3> RGBImgType;
	typedef Img<uint8_t, 1> DepthImgType;

	RGBImgType		m_imgRGB;
	DepthImgType	m_imgDepth;

	Channel<RGBImgType>		m_channelRGB;
	Channel<DepthImgType>	m_channelDepth;
};

///////////////////////////////////////////////////////////////////////////////

KinectGrabber::KinectGrabber()
	: Unit(Duration::milliseconds(100)),
	  m_listener(libfreenect2::Frame::Color |
				 libfreenect2::Frame::Ir |
				 libfreenect2::Frame::Depth),
	  m_undistortedDepth(512, 424, 4),
	  m_registeredRGB(512, 424, 4),
	  m_imgRGB(512, 424),
	  m_imgDepth(512, 424)
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

	m_registration = new libfreenect2::Registration(m_dev->getIrCameraParams(),
													m_dev->getColorCameraParams());

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &UnitName::onPoseChanged);
	m_channelRGB = publish<RGBImgType>("RGBImage");
	m_channelDepth = publish<DepthImgType>("DepthImage");
	for(auto dataIter = m_imgRGB.begin(); dataIter != m_imgRGB.end(); ++dataIter) {
		auto data = *dataIter;
		data[0] = 255;
		data[1] = 0;
		data[2] = 255;
	}
}

void KinectGrabber::process(const Timer& timer)
{
	if(!m_listener.waitForNewFrame(m_frames, 10)) { //m_pollTime.totalMilliseconds()))
		std::cout << "No new frames in this time frame" << std::endl;
		return;
	}

	libfreenect2::Frame* rgb = m_frames[libfreenect2::Frame::Color];
	//libfreenect2::Frame* ir = frames[libfreenect2::Frame::Ir];
	libfreenect2::Frame* depth = m_frames[libfreenect2::Frame::Depth];

	m_registration->apply(rgb, depth, &m_undistortedDepth, &m_registeredRGB);

	cv::Mat depthRaw(m_undistortedDepth.height, m_undistortedDepth.width, CV_8SC4);
	depthRaw.data = m_undistortedDepth.data;
	cv::Mat regDepthChannel[4];
	cv::split(depthRaw, regDepthChannel);
	cv::Mat tmpDepth;
	cv::flip(regDepthChannel[2], tmpDepth, 1);
	tmpDepth.convertTo(m_imgDepth, CV_8UC1, 1, 128);

	ChannelWrite<DepthImgType> wDepth = m_channelDepth.write();
	wDepth->value() = m_imgDepth;

	cv::Mat bgrxReg(m_registeredRGB.height, m_registeredRGB.width, CV_8UC4);
	bgrxReg.data = m_registeredRGB.data;
	cv::Mat tmpRGB(bgrxReg.rows, bgrxReg.cols, CV_8UC3);
	cv::cvtColor(bgrxReg, tmpRGB, CV_BGRA2BGR);
	cv::flip(tmpRGB, m_imgRGB, 1);

	ChannelWrite<RGBImgType> wRGB = m_channelRGB.write();
	wRGB->value() = m_imgRGB;

	m_listener.release(m_frames);
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(kinect::KinectGrabber, mira::Unit );
