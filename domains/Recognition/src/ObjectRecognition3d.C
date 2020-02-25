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

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/MicroUnit.h>
#include <image/Img.h>

#include <memory>

#include <Model.h>
#include <Tensor.h>

using namespace mira;

namespace recognition { 

///////////////////////////////////////////////////////////////////////////////

/**
 * Recognizes object in RGB images and locates them in 3D space
 */
class ObjectRecognition3d : public MicroUnit
{
MIRA_OBJECT(ObjectRecognition3d)

public:

	ObjectRecognition3d();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&ObjectRecognition3d::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &ObjectRecognition3d::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

private:

	typedef Img<uint8_t, 3> RGBImgType;
	typedef Img<uint8_t, 1> DepthImgType;

	void onNewRGBImage(ChannelRead<RGBImgType> image);
	void onNewDepthImage(ChannelRead<DepthImgType> image);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:

	Model					m_model;
	std::shared_ptr<Tensor>	m_input;
	std::shared_ptr<Tensor>	m_output;
	//Channel<Img<>> mChannel;
};

///////////////////////////////////////////////////////////////////////////////

ObjectRecognition3d::ObjectRecognition3d()
	: m_model("model.pb")
{
	// TODO: further initialization of members, etc.
}

void ObjectRecognition3d::initialize()
{
	subscribe<RGBImgType>("RGBImage", &ObjectRecognition3d::onNewRGBImage);
	subscribe<DepthImgType>("DepthImage", &ObjectRecognition3d::onNewDepthImage);

	m_model.init();

	m_input = std::make_shared<Tensor>(m_model, "input");
	m_output = std::make_shared<Tensor>(m_model, "output");

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &ObjectRecognition3d::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
}

void ObjectRecognition3d::onNewRGBImage(ChannelRead<ObjectRecognition3d::RGBImgType> image)
{
	std::vector<uint8_t> tmp(image->data(), image->data() + image->size().size()*3);
	m_input->set_data(tmp);
	m_model.run(*m_input, *m_output);
	for(float f : m_output->get_data<float>()) {
		std::cout << " " << f;
	}
	std::cout << std::endl;
}

void ObjectRecognition3d::onNewDepthImage(ChannelRead<ObjectRecognition3d::DepthImgType> image)
{

}

//void ObjectRecognition3d::onPoseChanged(ChannelRead<Pose2> data)
//{
	// TODO: this method is called whenever the pose has changed
//}

//void ObjectRecognition3d::setPose(const Pose2& pose)
//{
	// TODO: this can be called by RPC (by other authorities, by user from RPC Console/View, mirainspect, ...)
//}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
