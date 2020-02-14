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
 * @file FloatConsumer.C
 *    consumes floats
 *
 * @author Steffen Kastner
 * @date   2020/02/14
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/MicroUnit.h>

#include <deque>

using namespace mira;

namespace tutorials { 

///////////////////////////////////////////////////////////////////////////////

/**
 * consumes floats
 */
class FloatConsumer : public MicroUnit
{
MIRA_OBJECT(FloatConsumer)

public:

	FloatConsumer();

	template<typename Reflector>
	void reflect(Reflector& r)
	{
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&FloatConsumer::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &FloatConsumer::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

private:

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:

	//Channel<Img<>> mChannel;
	Channel<float>	m_meanChannel;
	float m_mean;

	std::deque<float> m_queue;

	void onNewData(ChannelRead<float> data);
};

///////////////////////////////////////////////////////////////////////////////

FloatConsumer::FloatConsumer()
{
	// TODO: further initialization of members, etc.
}

void FloatConsumer::initialize()
{
	subscribe<float>("FloatChannel", &FloatConsumer::onNewData);
	m_meanChannel = publish<float>("MeanChannel");
	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &FloatConsumer::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
}

void FloatConsumer::onNewData(ChannelRead<float> data)
{
	m_queue.push_back(data->value());
	if(m_queue.size() > 10)
		m_queue.pop_front();
	float sum = 0.f;
	std::for_each(m_queue.begin(), m_queue.end(), [&sum](float val){ sum += val; });
	m_mean = sum / m_queue.size();
	std::cout << "FloatConsumer: " << m_mean << std::endl;

	ChannelWrite<float> w = m_meanChannel.write();
	w->value() = m_mean;
}

//void FloatConsumer::onPoseChanged(ChannelRead<Pose2> data)
//{
	// TODO: this method is called whenever the pose has changed
//}

//void FloatConsumer::setPose(const Pose2& pose)
//{
	// TODO: this can be called by RPC (by other authorities, by user from RPC Console/View, mirainspect, ...)
//}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(tutorials::FloatConsumer, mira::MicroUnit);
