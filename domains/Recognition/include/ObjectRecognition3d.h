 
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

#include <recognitiondatatypes/Detection.h>
#include <kinectdatatypes/Types.h>

using namespace mira;

// forward declarations
namespace tensorflow {
	class Session;
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
	typedef kinectdatatypes::RGBImgType			RGBImgType;
	typedef kinectdatatypes::DepthImgType		DepthImgType;

	typedef recognitiondatatypes::Detection	Detection;

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

	cv::Point3f getXYZ(int r, int c, float depth);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:
	Channel<RGBImgType>		m_channelRGBMarked;
	Channel<Detection>		m_channelDetection;

	size_t	m_frameCount = 0;

	tf::Session*	m_session = nullptr;

	RegistrationData	m_regData;
	bool				m_hasRegData = false;

	DepthImgType		m_lastDepthImg;
};

} // namespace recognition

#endif // RECOGNITION_OBJECTRECKOGNITION3D_H
