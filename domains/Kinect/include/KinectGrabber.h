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
 * @file KinectGrabber.h
 *    Grabs RGB and depth images
 *
 * @author Steffen Kastner
 * @date   2019/10/01
 */

#ifndef KINECTGRABBER_H
#define KINECTGRABBER_H

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <fw/Unit.h>
#include <image/Img.h>

#include <exception>
#include <type_traits>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <kinectdatatypes/Types.h>

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
	typedef kinectdatatypes::RegistrationData	RegistrationData;
	typedef kinectdatatypes::RGBImgType			RGBImgType;
	typedef kinectdatatypes::DepthImgType		DepthImgType;
	typedef Img<uint8_t, 1>						DebugDepthImgType;

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
	size_t									m_frameNumber = 0;
	libfreenect2::Freenect2					m_freenect2;
	libfreenect2::Freenect2Device*			m_dev = nullptr;
	libfreenect2::Registration*				m_registration = nullptr;
	libfreenect2::SyncMultiFrameListener	m_listener;
	libfreenect2::FrameMap					m_frames;
	libfreenect2::Frame						m_undistortedDepth;
	libfreenect2::Frame						m_registeredRGB;
	libfreenect2::Frame						m_bigdepth;

	RegistrationData	m_regData;
	RGBImgType			m_imgRGB;
	RGBImgType			m_imgRGBFull;
	DepthImgType		m_imgDepth;
	DebugDepthImgType	m_imgDepthDebug;
	DepthImgType		m_imgDepthFull;
	DebugDepthImgType	m_imgDepthFullDebug;

	Channel<RegistrationData>	m_channelRegistrationData;
	Channel<RGBImgType>			m_channelRGB;
	Channel<RGBImgType>			m_channelRGBFull;
	Channel<DepthImgType>		m_channelDepth;
	Channel<DebugDepthImgType>	m_channelDepthDebug;
	Channel<DepthImgType>		m_channelDepthFull;
	Channel<DebugDepthImgType>	m_channelDepthFullDebug;
};

} // namespace kinect

#endif // KINECTGRABBER_H
