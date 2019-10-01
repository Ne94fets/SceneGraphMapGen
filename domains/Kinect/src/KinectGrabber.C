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

#include <fw/Unit.h>

#include <exception>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

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
		Unit::reflect(r);

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
	Duration	m_pollTime = Duration::milliseconds(100);

	libfreenect2::Freenect2					m_freenect2;
	libfreenect2::Freenect2Device*			m_dev = nullptr;
	libfreenect2::PacketPipeline*			m_pipeline = nullptr;
	//libfreenect2::Registration*				m_registration = nullptr;
	libfreenect2::SyncMultiFrameListener	m_listener;
	libfreenect2::FrameMap					m_frames;
	libfreenect2::Frame						m_undistortedRGB;
	libfreenect2::Frame						m_registeredDepth;

	//Channel<Img<>> mChannel;
};

///////////////////////////////////////////////////////////////////////////////

KinectGrabber::KinectGrabber()
	: Unit(m_pollTime),
	  m_listener(libfreenect2::Frame::Color |
				 libfreenect2::Frame::Ir |
				 libfreenect2::Frame::Depth),
	  m_undistortedRGB(512, 424, 4),
	  m_registeredDepth(512, 424, 4)
{
	m_pipeline = new libfreenect2::CpuPacketPipeline();
	if(m_freenect2.enumerateDevices() == 0)
		throw std::runtime_exception("No Kinect connected!");
	std::string serial = m_freenect2.getDefaultDeviceSerialNumber();
	m_dev = m_freenect2.openDevice(serial, m_pipeline);
	m_dev->setColorFrameListener(&m_listener);
	m_dev->setIrAndDepthFrameListener(&m_listener);
}

KinectGrabber::~KinectGrabber()
{
	m_dev->stop();
	m_dev->close();
	delete m_dev;
	delete m_pipeline;
}

void KinectGrabber::initialize()
{
	if(!m_dev->start())
		throw std::runtime_exception("Could not start Kinect");
	if(!m_dev->startStreams(true, true))
		throw std::runtime_exception("Could not start Kinect Streams");
//	m_registration = new libfreenect2::Registration(
//				m_dev->getIrCameraParams(),
//				m_dev->getColorCameraParams());

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &UnitName::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
}

void KinectGrabber::process(const Timer& timer)
{
	if(!m_listener.waitForNewFrame(m_frames, m_pollTime.count()))
		return;

	libfreenect2::Frame* rgb = m_frames[libfreenect2::Frame::Color];
	//libfreenect2::Frame* ir = frames[libfreenect2::Frame::Ir];
	libfreenect2::Frame* depth = m_frames[libfreenect2::Frame::Depth];

	//m_registration->apply(rgb, depth, &m_undistortedRGB, &m_registeredDepth);

	m_listener.release(m_frames);
	// TODO: this method is called periodically with the specified cycle time, so you can perform your computation here.
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(kinect::KinectGrabber, mira::Unit );
