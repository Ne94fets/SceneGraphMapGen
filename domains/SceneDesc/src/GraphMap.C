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
 * @file GraphMap.C
 *    Graph-based scene description
 *
 * @author Steffen Kastner
 * @date   2020/04/11
 */

//#include <transform/Pose.h> // TODO: enable to use Pose2!
#include <fw/MicroUnit.h>

#include <neo4j-client.h>

#include "recognitiondatatypes/Detection.h"

using namespace mira;

namespace scenedesc {

///////////////////////////////////////////////////////////////////////////////

/**
 * Graph-based scene description
 */
class GraphMap : public MicroUnit {
	MIRA_OBJECT(GraphMap)

public:

	typedef recognitiondatatypes::Detection	Detection;

public:

	GraphMap();
	~GraphMap();

	template<typename Reflector>
	void reflect(Reflector& r) {
		MIRA_REFLECT_BASE(r, MicroUnit);

		// TODO: reflect all parameters (members and properties) that specify the persistent state of the unit
		//r.property("Param1", mParam1, "First parameter of this unit with default value", 123.4f);
		//r.member("Param2", mParam2, setter(&GraphMap::setParam2,this), "Second parameter with setter");

		// TODO: reflect all methods if this is a service providing RPCs
		//r.method("setPose", &GraphMap::setPose, this, "Set the pose",
		//         "pose", "The pose to set", Pose2(1.0, 2.0, deg2rad(90.0)));
	}

protected:

	virtual void initialize();

private:

	void onObjectDetection(ChannelRead<Detection> detection);
	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:
	neo4j_connection_t* m_connection = nullptr;

	//Channel<Img<>> mChannel;
};

///////////////////////////////////////////////////////////////////////////////

GraphMap::GraphMap() {
	// TODO: further initialization of members, etc.
	neo4j_client_init();
}

GraphMap::~GraphMap() {
	if(m_connection)
		neo4j_close(m_connection);

	neo4j_client_cleanup();
}

void GraphMap::initialize() {
	m_connection = neo4j_connect("bolt://neo4j:neopassword4j@localhost:7687", nullptr, NEO4J_INSECURE);
	if(!m_connection) {
		char buf[128];
		const char* msg = neo4j_strerror(errno, buf, sizeof(buf));
		std::stringstream ss;
		ss << "Could not connect to neo4j database: ";
		ss << msg;
		throw std::runtime_error(ss.str());
	}

	subscribe<Detection>("ObjectDetection", &GraphMap::onObjectDetection);
}

void GraphMap::onObjectDetection(ChannelRead<GraphMap::Detection> detection) {
	Detection d = *detection;
	std::string typeName = Detection::getName(d.type);
	std::cout << "Detected " << typeName << std::endl;

	neo4j_result_stream_t* results = neo4j_run(m_connection, "RETURN 'hello world'", neo4j_null);
	if(!results)
		throw std::runtime_error("Could not execute neo4j statement");

	neo4j_result_t* result = neo4j_fetch_next(results);
	if(!result)
		throw std::runtime_error("Could not fetch neo4j result");

	neo4j_value_t value = neo4j_result_field(result, 0);
	char buf[128];
	neo4j_tostring(value, buf, sizeof(buf));
	std::cout << buf << std::endl;

	neo4j_close_results(results);
}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(scenedesc::GraphMap, mira::MicroUnit);
