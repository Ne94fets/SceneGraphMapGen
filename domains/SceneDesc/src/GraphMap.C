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
#include <filter/ChannelSynchronizerSequenceID.h>

#include <chrono>
#include <thread>

#include <neo4j-client.h>

#include "kinectdatatypes/RGBDQueue.h"
#include "kinectdatatypes/Types.h"
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
	typedef recognitiondatatypes::Detection				Detection;
	typedef recognitiondatatypes::DetectionContainer	DetectionContainer;

	typedef Eigen::Matrix4f	TransformType;

	typedef ChannelSynchronizerSequenceID3<DetectionContainer, DetectionContainer, TransformType>	SyncQueue;


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

	void onSynchronized(ChannelRead<DetectionContainer> detections,
						ChannelRead<DetectionContainer> detectionsNew,
						ChannelRead<TransformType> globalPose);

	void analyseDetections(const DetectionContainer& detections,
						   const DetectionContainer& detectionsNew,
						   const TransformType& globalTransform);

	bool existsObject(const Detection& d, const boost::uuids::uuid& room);
	bool existsObjectDBuuid(const Detection& d);
	bool existsObjectWithin(double minx, double maxx, double miny, double maxy, double minz, double maxz, const boost::uuids::uuid& room, const Detection& d);
	void addRoom(const boost::uuids::uuid& room);
	void addObject(const Detection& d, const boost::uuids::uuid& room);
	void addRelation(const Detection& d, const Detection& other, cv::Point3f& relativeOffset);

private:
	volatile bool	m_shutdown = false;

	neo4j_connection_t* m_connection = nullptr;
	boost::uuids::uuid	m_room;

	SyncQueue						m_syncQueue;

	std::thread*	m_worker = nullptr;

	//Channel<Img<>> mChannel;
};

///////////////////////////////////////////////////////////////////////////////

GraphMap::GraphMap() {
	// TODO: further initialization of members, etc.
	neo4j_client_init();
	m_room = boost::uuids::random_generator()();
}

GraphMap::~GraphMap() {
	m_shutdown = true;

	if(m_worker) {
		m_worker->join();
		delete m_worker;
	}

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

	// add first room
	addRoom(m_room);

	m_syncQueue.subscribe(*this,
						  "ObjectDetection",
						  "ObjectDetectionNew",
						  "PCGlobalTransform",
						  &GraphMap::onSynchronized,
						  this,
						  Duration::seconds(10));
}

void GraphMap::onSynchronized(ChannelRead<GraphMap::DetectionContainer> detections, ChannelRead<GraphMap::DetectionContainer> detectionsNew, ChannelRead<GraphMap::TransformType> globalPose) {
	analyseDetections(detections, detectionsNew, globalPose);
}

void GraphMap::analyseDetections(const DetectionContainer& detections,
								 const DetectionContainer& detectionsNew,
								 const TransformType& globalTransform) {
	if(detections.empty() || detectionsNew.empty())
		return;

	std::vector<Detection> transformedDetections;
	transformedDetections.reserve(detections.size());
	std::vector<Detection> transformedDetectionsNew;
	transformedDetectionsNew.reserve(detectionsNew.size());

	// transform detections into world coordinate system
	// only those who are not new detections
	for(const auto& d : detections) {
		if(std::isnan(d.pos.x)) {
			continue;
		}
		bool isNew = false;
		for(const auto& dnew : detectionsNew) {
			if(d.uuid == dnew.uuid) {
				isNew = true;
				break;
			}
		}
		if(isNew) {
			continue;
		}

		Detection tDet = d;
		Eigen::Vector4f pos(d.pos.x, d.pos.y, d.pos.z, 1);
		pos = globalTransform * pos;
		tDet.pos = cv::Point3f(pos.x(), pos.y(), pos.z());
		transformedDetections.push_back(tDet);
	}

	for(const auto& d : detectionsNew) {
		if(std::isnan(d.pos.x)) {
			continue;
		}

		Detection tDet = d;
		Eigen::Vector4f pos(d.pos.x, d.pos.y, d.pos.z, 1);
		pos = globalTransform * pos;
		tDet.pos = cv::Point3f(pos.x(), pos.y(), pos.z());
		transformedDetectionsNew.push_back(tDet);
	}

	// insert only missing objects
	for(size_t i = 0; i < transformedDetectionsNew.size(); ++i) {
		const auto& d0 = transformedDetectionsNew[i];

		// skip object if already existing
		if(existsObject(d0, m_room)) { continue; }

		addObject(d0, m_room);

		// calc relations to object if other exists
		for(size_t j = 0; j < transformedDetections.size(); ++j) {
			const auto& d1 = transformedDetections[j];

			cv::Point3f rel = d1.pos - d0.pos;
			addRelation(d0, d1, rel);
		}

		// put new detection into detections
		transformedDetections.push_back(d0);
	}
}

bool GraphMap::existsObject(const GraphMap::Detection& d, const boost::uuids::uuid& room) {
	if(existsObjectDBuuid(d)) {
		return true;
	}

	float minx = d.pos.x + d.bboxMin.x;
	float maxx = d.pos.x + d.bboxMax.x;
	float miny = d.pos.y + d.bboxMin.y;
	float maxy = d.pos.y + d.bboxMax.y;
	float minz = d.pos.z + d.bboxMin.z;
	float maxz = d.pos.z + d.bboxMax.z;
	return existsObjectWithin(minx, maxx, miny, maxy, minz, maxz, room, d);
}

bool GraphMap::existsObjectDBuuid(const GraphMap::Detection& d) {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];
	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = neo4j_run(m_connection, "MATCH (object:object "
															 "{uuid: $uuid}) "
															 "RETURN object", params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_result_t* result = neo4j_fetch_next(results);
	bool hasResult = true;
	if(!result) {
		hasResult = false;
	}

	neo4j_close_results(results);

	return hasResult;
}

bool GraphMap::existsObjectWithin(double minx, double maxx,
								  double miny, double maxy,
								  double minz, double maxz,
								  const boost::uuids::uuid& room,
								  const Detection& d) {
	const unsigned int numEntries = 8;
	neo4j_map_entry_t entries[numEntries];

	entries[0] = neo4j_map_kentry(neo4j_string("min_x"),
								  neo4j_float(minx));
	entries[1] = neo4j_map_kentry(neo4j_string("min_y"),
								  neo4j_float(miny));
	entries[2] = neo4j_map_kentry(neo4j_string("min_z"),
								  neo4j_float(minz));
	entries[3] = neo4j_map_kentry(neo4j_string("max_x"),
								  neo4j_float(maxx));
	entries[4] = neo4j_map_kentry(neo4j_string("max_y"),
								  neo4j_float(maxy));
	entries[5] = neo4j_map_kentry(neo4j_string("max_z"),
								  neo4j_float(maxz));

	std::string roomUUID = boost::uuids::to_string(room);
	entries[6] = neo4j_map_kentry(neo4j_string("room_uuid"),
								  neo4j_ustring(roomUUID.c_str(),
												static_cast<unsigned int>(roomUUID.length())));

	std::string detectionType = Detection::getTypeName(d.type);
	entries[7] = neo4j_map_kentry(neo4j_string("type"),
								  neo4j_ustring(detectionType.c_str(),
												static_cast<unsigned int>(detectionType.length())));

	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = neo4j_run(m_connection, "MATCH (r:room {uuid: $room_uuid})"
															 "-[:CONTAINS]->(o:object {name: $type}) "
															 "WHERE "
															 "$min_x < o.pos_x AND o.pos_x < $max_x AND "
															 "$min_y < o.pos_y AND o.pos_y < $max_y AND "
															 "$min_z < o.pos_z AND o.pos_z < $max_z "
															 "RETURN o", params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_result_t* result = neo4j_fetch_next(results);
	bool hasResult = true;
	if(!result) {
		hasResult = false;
	}

	neo4j_close_results(results);

	return hasResult;
}

void GraphMap::addRoom(const boost::uuids::uuid& room) {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(room);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));

	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = neo4j_send(m_connection, "CREATE (:room {"
															  "uuid: $uuid, "
															  "name: 'Room'})", params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_close_results(results);
}

void GraphMap::addObject(const GraphMap::Detection& d, const boost::uuids::uuid& room) {
	const unsigned int numEntries = 6;
	neo4j_map_entry_t entries[numEntries];
	std::string detectionType = Detection::getTypeName(d.type);

	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	entries[1] = neo4j_map_kentry(neo4j_string("type"),
								  neo4j_ustring(detectionType.c_str(),
												static_cast<unsigned int>(detectionType.length())));
	std::string roomUUID = boost::uuids::to_string(room);
	entries[2] = neo4j_map_kentry(neo4j_string("room_uuid"),
								  neo4j_ustring(roomUUID.c_str(),
												static_cast<unsigned int>(roomUUID.length())));

	entries[3] = neo4j_map_kentry(neo4j_string("pos_x"),
								  neo4j_float(static_cast<double>(d.pos.x)));
	entries[4] = neo4j_map_kentry(neo4j_string("pos_y"),
								  neo4j_float(static_cast<double>(d.pos.y)));
	entries[5] = neo4j_map_kentry(neo4j_string("pos_z"),
								  neo4j_float(static_cast<double>(d.pos.z)));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = neo4j_send(m_connection, "MATCH (r:room) "
															  "WHERE r.uuid = $room_uuid "
															  "CREATE (o:object {"
															  "uuid: $uuid, "
															  "name: $type, "
															  "pos_x: $pos_x, "
															  "pos_y: $pos_y, "
															  "pos_z: $pos_z})"
															  "CREATE (r)"
															  "-[:CONTAINS]->"
															  "(o)", params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_close_results(results);
}

void GraphMap::addRelation(
		const GraphMap::Detection& d,
		const GraphMap::Detection& other,
		cv::Point3f& relativeOffset) {
	const unsigned int numEntries = 6;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid_d"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));

	std::string uuidOtherStr = boost::uuids::to_string(other.uuid);
	entries[1] = neo4j_map_kentry(neo4j_string("uuid_other"),
								  neo4j_ustring(uuidOtherStr.c_str(),
												static_cast<unsigned int>(uuidOtherStr.length())));
	entries[2] = neo4j_map_kentry(neo4j_string("rel_x"),
								  neo4j_float(static_cast<double>(relativeOffset.x)));
	entries[3] = neo4j_map_kentry(neo4j_string("rel_y"),
								  neo4j_float(static_cast<double>(relativeOffset.y)));
	entries[4] = neo4j_map_kentry(neo4j_string("rel_z"),
								  neo4j_float(static_cast<double>(relativeOffset.z)));
	entries[5] = neo4j_map_kentry(neo4j_string("distance"),
								  neo4j_float(static_cast<double>(std::sqrt(relativeOffset.dot(relativeOffset)))));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = neo4j_send(m_connection, "MATCH (a:object), (b:object) "
															  "WHERE a.uuid = $uuid_d AND b.uuid = $uuid_other "
															  "CREATE (a)"
															  "-[:OFFSET {"
															  "x: $rel_x, "
															  "y: $rel_y, "
															  "z: $rel_z, "
															  "distance: $distance}]->"
															  "(b)", params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_close_results(results);

}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(scenedesc::GraphMap, mira::MicroUnit);
