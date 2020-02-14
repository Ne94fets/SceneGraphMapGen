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
 * @file FloatProducer.C
 *    produces floats
 *
 * @author Steffen Kastner
 * @date   2020/02/14
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/Unit.h>

using namespace mira;

namespace tutorials { 

///////////////////////////////////////////////////////////////////////////////

/**
 * produces floats
 */
class FloatProducer : public Unit
{
MIRA_OBJECT(FloatProducer)

public:

	FloatProducer();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, Unit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&FloatProducer::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &FloatProducer::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

	virtual void process(const Timer& timer);

private:

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:

	//Channel<Img<>> mChannel;
	Channel<float> m_FloatChannel;
	float m_value = 0.f;
};

///////////////////////////////////////////////////////////////////////////////

FloatProducer::FloatProducer() : Unit(Duration::milliseconds(100))
{
	// TODO: further initialization of members, etc.
}

void FloatProducer::initialize()
{
	m_FloatChannel = publish<float>("FloatChannel");
	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &FloatProducer::onPoseChanged);
	//mChannel = publish<Img<>>("Image");

	// TODO: publish service to enable RPCs
	//publishService(*this);
}

void FloatProducer::process(const Timer& timer)
{
	m_value += 1.f;
	m_FloatChannel.post(m_value);
	std::cout << "FloatProducer: " << m_value << std::endl;
	// TODO: this method is called periodically with the specified cycle time, so you can perform your computation here.
}

//void FloatProducer::onPoseChanged(ChannelRead<Pose2> data)
//{
	// TODO: this method is called whenever the pose has changed
//}

//void FloatProducer::setPose(const Pose2& pose)
//{
	// TODO: this can be called by RPC (by other authorities, by user from RPC Console/View, mirainspect, ...)
//}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(tutorials::FloatProducer, mira::Unit);
