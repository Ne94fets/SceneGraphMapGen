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

	typedef std::function<void(const Detection&, const std::string&, neo4j_connection*)>	DBFunType;
	typedef std::unordered_map<std::string, DBFunType>										DBFunMap;

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
	bool existsObjectUuid(const Detection& d);
	bool existsObjectWithin(double minx, double maxx, double miny, double maxy, double minz, double maxz, const boost::uuids::uuid& room, const Detection& d);
	void addLogic();
	void addRoom(const boost::uuids::uuid& room);
	void addObject(const Detection& d, const boost::uuids::uuid& room);
	void addRelation(const Detection& d, const Detection& other, cv::Point3f& relativeOffset);
	void addAttibutes(const Detection& d);
	std::string genRoomDescription() const;
	std::string genRoomObjCountDescription(const std::string& uuid) const;
	std::string genObjectDescription(const std::string& uuid, const Eigen::Vector3f& center) const;
	std::string genAttributeDescription(const std::string& name, const std::string& type) const;
	std::string genDescriptionFromOffsetRelation(const std::string& uuid,
												 const std::string& name,
												 const Eigen::Vector3f& center) const;
	Eigen::Vector3f getRoomCenter() const;

	static void execInsertQuery(neo4j_connection_t* conn,
								const std::string& query,
								const neo4j_value_t& params);
	static bool execExistsQuery(neo4j_connection_t* conn,
								const std::string& query,
								const neo4j_value_t& params);
	static neo4j_result_stream_t* execResultQuery(neo4j_connection_t* conn,
												  const std::string& query,
												  const neo4j_value_t& params);

	static std::string color2Str(const cv::Scalar& c);
	static std::string dir2Str(const Eigen::Vector3f& watchDir, const Eigen::Vector3f& offset);

	static void insertColor(const Detection& d, const std::string& attrType, neo4j_connection_t* conn);
	static void insertState(const Detection& d, const std::string& attrType, neo4j_connection_t* conn);
	static void insertScreenSize(const Detection& d, const std::string& attrType, neo4j_connection_t* conn);

private:
	volatile bool	m_shutdown = false;

	neo4j_connection_t* m_connection = nullptr;
	boost::uuids::uuid	m_room;

	SyncQueue						m_syncQueue;

	std::thread*	m_worker = nullptr;

	std::vector<long> m_durations;

	static DBFunMap dbFunMap;
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

	// add some logic if not present
	addLogic();

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
	auto startTime = std::chrono::system_clock::now();

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
		addAttibutes(d0);

		// calc relations to object if other exists
		for(size_t j = 0; j < transformedDetections.size(); ++j) {
			const auto& d1 = transformedDetections[j];

			cv::Point3f rel = d1.pos - d0.pos;
			addRelation(d0, d1, rel);
		}

		// put new detection into detections
		transformedDetections.push_back(d0);
	}

	auto endTime = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	if(duration > 1000/30) {
		std::cout << "GraphMap analyse took: " << duration << "ms" << std::endl;
	}

	m_durations.push_back(duration);
	float avg(0);
	for(const auto d : m_durations) {
		avg += float(d);
	}
	avg /= m_durations.size();
	std::cout << "GraphMap: Average insert duration: " << avg << std::endl;

	std::cout << genRoomDescription() << std::endl;
}

bool GraphMap::existsObject(const GraphMap::Detection& d, const boost::uuids::uuid& room) {
	if(existsObjectUuid(d)) {
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

bool GraphMap::existsObjectUuid(const GraphMap::Detection& d) {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];
	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	return execExistsQuery(m_connection,
						   "MATCH (object:object "
						   "{uuid: $uuid}) "
						   "RETURN object", params);
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
	return execExistsQuery(m_connection,
						   "MATCH (r:room {uuid: $room_uuid})"
						   "-[:CONTAINS]->(o:object {name: $type}) "
						   "WHERE "
						   "$min_x < o.pos_x AND o.pos_x < $max_x AND "
						   "$min_y < o.pos_y AND o.pos_y < $max_y AND "
						   "$min_z < o.pos_z AND o.pos_z < $max_z "
						   "RETURN o",
						   params);
}

void GraphMap::addLogic() {
	neo4j_value_t params = neo4j_map(nullptr, 0);

	if(execExistsQuery(m_connection,
					   "MATCH (l:logic) WHERE l.name = 'Logic' RETURN l",
					   params)) {
		return;
	}

	execInsertQuery(m_connection,
					"CREATE (l:logic {name: 'Logic'}) "
					"CREATE (m:logic {name: 'Map'}) "
					"CREATE (ot:logic {name: 'ObjectType'}) "
					"CREATE (tv:logic {name: 'tv'}) "
					"CREATE (tvState:logic {name: 'State'}) "
					"CREATE (tvStateIN:logic {name: 'insertState'}) "
					"CREATE (tvScreenSize:logic {name: 'ScreenSize'}) "
					"CREATE (tvScreenSizeIN:logic {name: 'insertScreenSize'}) "
					"CREATE (chair:logic {name: 'chair'}) "
					"CREATE (chairColor:logic {name: 'Color'}) "
					"CREATE (chairColorIN:logic {name: 'insertColor'}) "
					"CREATE (mouse:logic {name: 'mouse'}) "
					"CREATE (mouseColor:logic {name: 'Color'}) "
					"CREATE (mouseColorIN:logic {name: 'insertColor'}) "
					"CREATE "
					"(l)-[:FOR]->(m), "
					"(m)-[:FOR]->(ot), "
					"(ot)-[:FOR]->(tv), "
					"(tv)-[:ATTRIBUTE]->(tvState), "
					"(tvState)-[:INSERT]->(tvStateIN), "
					"(tv)-[:ATTRIBUTE]->(tvScreenSize), "
					"(tvScreenSize)-[:INSERT]->(tvScreenSizeIN), "
					"(ot)-[:FOR]->(chair), "
					"(chair)-[:ATTRIBUTE]->(chairColor), "
					"(chairColor)-[:INSERT]->(chairColorIN), "
					"(ot)-[:FOR]->(mouse), "
					"(mouse)-[:ATTRIBUTE]->(mouseColor), "
					"(mouseColor)-[:INSERT]->(mouseColorIN)",
					params);
}

void GraphMap::addRoom(const boost::uuids::uuid& room) {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(room);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));

	neo4j_value_t params = neo4j_map(entries, numEntries);
	execInsertQuery(m_connection,
					"CREATE (:room {"
					"uuid: $uuid, "
					"name: 'Room'})",
					params);
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
	execInsertQuery(m_connection,
					"MATCH (r:room) "
					"WHERE r.uuid = $room_uuid "
					"CREATE (o:object {"
					"uuid: $uuid, "
					"name: $type, "
					"pos_x: $pos_x, "
					"pos_y: $pos_y, "
					"pos_z: $pos_z})"
					"CREATE (r)"
					"-[:CONTAINS]->"
					"(o)",
					params);
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
	execInsertQuery(m_connection,
					"MATCH (a:object), (b:object) "
					"WHERE a.uuid = $uuid_d AND b.uuid = $uuid_other "
					"CREATE (a)"
					"-[:OFFSET {"
					"x: $rel_x, "
					"y: $rel_y, "
					"z: $rel_z, "
					"distance: $distance}]->"
					"(b)",
					params);
}

void GraphMap::addAttibutes(const GraphMap::Detection& d) {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	std::string typeStr = Detection::getTypeName(d.type);
	entries[0] = neo4j_map_kentry(neo4j_string("obj_type"),
								  neo4j_ustring(typeStr.c_str(),
												static_cast<unsigned int>(typeStr.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH "
				"(:logic {name: 'Logic'})-[:FOR]->"
				"(:logic {name: 'Map'})-[:FOR]->"
				"(:logic {name: 'ObjectType'})-[:FOR]->"
				"(:logic {name: $obj_type})-[:ATTRIBUTE]->"
				"(a:logic)-[:INSERT]->(fn:logic) "
				"RETURN a.name, fn.name",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	while(result) {
		neo4j_value_t attrVal = neo4j_result_field(result, 0);
		neo4j_value_t fnNameVal = neo4j_result_field(result, 1);
		char buf[64];
		neo4j_tostring(attrVal, buf, sizeof(buf));
		std::string attrName(buf+1, std::strlen(buf)-2);
		neo4j_tostring(fnNameVal, buf, sizeof(buf));
		std::string fnName(buf+1, std::strlen(buf)-2);

		const auto& fnIter = dbFunMap.find(fnName);
		if(fnIter != dbFunMap.end()) {
			// execute function
			fnIter->second(d, attrName, m_connection);
		}

		result = neo4j_fetch_next(results);
	}
}

std::string GraphMap::genRoomDescription() const {
	Eigen::Vector3f center = getRoomCenter();

	std::stringstream description;

	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	std::string room_uuid = boost::uuids::to_string(m_room);
	entries[0] = neo4j_map_kentry(neo4j_string("room_uuid"),
								  neo4j_ustring(room_uuid.c_str(),
												static_cast<unsigned int>(room_uuid.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH (r:room {uuid:$room_uuid})-[:CONTAINS]->(o:object) "
				"RETURN o.uuid, o.name",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	if(!result) {
		description << "The room contains nothing. ";
	}

	description << genRoomObjCountDescription(room_uuid);

	size_t cnt = 0;
	while(result) {
		neo4j_value_t o_uuid = neo4j_result_field(result, 0);
		neo4j_value_t o_name = neo4j_result_field(result, 1);
		char buf[64];
		neo4j_tostring(o_uuid, buf, sizeof(buf));
		std::string o_uuid_str(buf+1, std::strlen(buf)-2);
		neo4j_tostring(o_name, buf, sizeof(buf));
		std::string o_name_str(buf+1, std::strlen(buf)-2);

		if(cnt == 0) { description << "One object is a "; }
		else { description << "Another object this room contains is a "; }
		description << o_name_str << ". ";
		description << genObjectDescription(o_uuid_str, center);

		result = neo4j_fetch_next(results);
		cnt++;
	}

	neo4j_close_results(results);

	return description.str();
}

std::string GraphMap::genRoomObjCountDescription(const std::string& uuid) const {
	std::stringstream description;

	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	entries[0] = neo4j_map_kentry(neo4j_string("room_uuid"),
								  neo4j_ustring(uuid.c_str(),
												static_cast<unsigned int>(uuid.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH (r:room {uuid:$room_uuid})-[:CONTAINS]->(o:object) "
				"RETURN count(o)",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	neo4j_value_t objCnt = neo4j_result_field(result, 0);
	auto objCntVal = neo4j_int_value(objCnt);
	description << "The room contains " << objCntVal << " object";
	if(objCntVal > 1) { description << "s"; }

	neo4j_close_results(results);
	results = execResultQuery(
				m_connection,
				"MATCH (r:room {uuid:$room_uuid})-[:CONTAINS]->(o:object) "
				"WITH DISTINCT o.name as ns "
				"RETURN count(ns)",
				params);

	result = neo4j_fetch_next(results);
	neo4j_value_t typeCnt = neo4j_result_field(result, 0);
	auto typeCntVal = neo4j_int_value(typeCnt);
	description << " of " << typeCntVal << " distinct categor";
	if(objCntVal > 1) { description << "ies"; }
	else { description << "y"; }
	description << ". ";

	return description.str();
}

std::string GraphMap::genObjectDescription(const std::string& uuid,
										   const Eigen::Vector3f& center) const {
	std::stringstream description;

	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	entries[0] = neo4j_map_kentry(neo4j_string("obj_uuid"),
								  neo4j_ustring(uuid.c_str(),
												static_cast<unsigned int>(uuid.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH (o:object {uuid:$obj_uuid})-[:IS]->(a:attribute) "
				"RETURN o.name, a.type, a.name",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	std::vector<std::tuple<std::string, std::string, std::string>> attrs;
	while(result) {
		neo4j_value_t o_name = neo4j_result_field(result, 0);
		neo4j_value_t a_type = neo4j_result_field(result, 1);
		neo4j_value_t a_name = neo4j_result_field(result, 2);
		char buf[64];
		neo4j_tostring(o_name, buf, sizeof(buf));
		std::string o_name_str(buf+1, std::strlen(buf)-2);
		neo4j_tostring(a_type, buf, sizeof(buf));
		std::string a_type_str(buf+1, std::strlen(buf)-2);
		neo4j_tostring(a_name, buf, sizeof(buf));
		std::string a_name_str(buf+1, std::strlen(buf)-2);

		attrs.push_back({o_name_str, a_type_str, a_name_str});

		result = neo4j_fetch_next(results);
	}

	neo4j_close_results(results);

	if(attrs.empty()) {
		return description.str();
	}

	std::string attrdesc = genAttributeDescription(std::get<2>(attrs[0]), std::get<1>(attrs[0]));
	description << "The " << std::get<0>(attrs[0]) << " " << attrdesc;
	if(attrs.size() > 2) {
		for(size_t i = 1; i < attrs.size()-1; ++i) {
			std::string attrdesc = genAttributeDescription(std::get<2>(attrs[i]), std::get<1>(attrs[i]));
			description << ", " << attrdesc;
		}
		std::string attrdesc = genAttributeDescription(std::get<2>(attrs.back()), std::get<1>(attrs.back()));
		description << " and " << attrdesc << ". ";
	} else if(attrs.size() > 1) {
		std::string attrdesc = genAttributeDescription(std::get<2>(attrs[1]), std::get<1>(attrs[1]));
		description << " and " << attrdesc << ". ";
	} else {
		description << ". ";
	}

	std::string name = std::get<0>(attrs[0]);
	description << genDescriptionFromOffsetRelation(uuid, name, center);

	return description.str();
}

std::string GraphMap::genAttributeDescription(const std::string& name,
											  const std::string& type) const {
	std::stringstream description;
	if(type == "Color") {
		description << "is " << name;
	} else if(type == "State") {
		description << "is " << name;
	} else if(type == "ScreenSize") {
		description << "has " << name;
	}
	return description.str();
}

std::string GraphMap::genDescriptionFromOffsetRelation(const std::string& uuid,
													   const std::string& name,
													   const Eigen::Vector3f& center) const {
	std::stringstream description;

	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	entries[0] = neo4j_map_kentry(neo4j_string("obj_uuid"),
								  neo4j_ustring(uuid.c_str(),
												static_cast<unsigned int>(uuid.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH (o:object {uuid:$obj_uuid})-[off:OFFSET]->(oo:object) "
				"RETURN o.pos_x, o.pos_y, o.pos_z, off.x, off.y, off.z, oo.name",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	std::map<std::string, std::vector<std::string>> dirs;
	while(result) {
		neo4j_value_t pos_x = neo4j_result_field(result, 0);
		neo4j_value_t pos_y = neo4j_result_field(result, 1);
		neo4j_value_t pos_z = neo4j_result_field(result, 2);
		neo4j_value_t off_x = neo4j_result_field(result, 3);
		neo4j_value_t off_y = neo4j_result_field(result, 4);
		neo4j_value_t off_z = neo4j_result_field(result, 5);
		neo4j_value_t oo_name = neo4j_result_field(result, 6);
		char buf[64];
		neo4j_tostring(oo_name, buf, sizeof(buf));
		std::string oo_name_str(buf+1, std::strlen(buf)-2);
		Eigen::Vector3f objPos(0,0,0);
		objPos.x() = neo4j_float_value(pos_x);
		objPos.y() = neo4j_float_value(pos_y);
		Eigen::Vector3f offset;
		offset.x() = neo4j_float_value(off_x);
		offset.y() = neo4j_float_value(off_y);
		offset.z() = neo4j_float_value(off_z);
		auto c2o = objPos - center;

		dirs[dir2Str(c2o, offset)].push_back(oo_name_str);

		result = neo4j_fetch_next(results);
	}

	neo4j_close_results(results);


	for(auto& pair : dirs) {
		auto& nObjects = pair.second;
		if(nObjects.empty()) {
			continue;
		}

		description << "The " << name << " is " << pair.first << " a " << nObjects.back();
		nObjects.pop_back();
		while(nObjects.size() > 1) {
			description << ", a " << nObjects.back();
			nObjects.pop_back();
		}
		if(nObjects.size() == 1) {
			description << " and a " << nObjects.back();
			nObjects.pop_back();
		}
		description << ". ";
	}

	return description.str();
}

Eigen::Vector3f GraphMap::getRoomCenter() const {
	const unsigned int numEntries = 1;
	neo4j_map_entry_t entries[numEntries];

	std::string room_uuid = boost::uuids::to_string(m_room);
	entries[0] = neo4j_map_kentry(neo4j_string("room_uuid"),
								  neo4j_ustring(room_uuid.c_str(),
												static_cast<unsigned int>(room_uuid.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	neo4j_result_stream_t* results = execResultQuery(
				m_connection,
				"MATCH (r:room {uuid:$room_uuid})-[:CONTAINS]->(o:object) "
				"RETURN o.pos_x, o.pos_y, o.pos_z",
				params);

	neo4j_result_t* result = neo4j_fetch_next(results);
	Eigen::Vector3f center(0,0,0);
	size_t cnt = 0;
	while(result) {
		neo4j_value_t pos_x = neo4j_result_field(result, 0);
		neo4j_value_t pos_y = neo4j_result_field(result, 1);
		neo4j_value_t pos_z = neo4j_result_field(result, 2);
		center.x() += neo4j_float_value(pos_x);
		center.y() += neo4j_float_value(pos_y);
		center.z() += neo4j_float_value(pos_z);
		cnt++;

		result = neo4j_fetch_next(results);
	}

	neo4j_close_results(results);

	center /= cnt;
	return center;
}

void GraphMap::execInsertQuery(neo4j_connection_t* conn,
							   const std::string& query,
							   const neo4j_value_t& params) {
	neo4j_result_stream_t* results = neo4j_send(conn, query.c_str(), params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	neo4j_close_results(results);
}

bool GraphMap::execExistsQuery(neo4j_connection_t* conn,
							   const std::string& query,
							   const neo4j_value_t& params) {
	neo4j_result_stream_t* results = neo4j_run(conn, query.c_str(), params);

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

neo4j_result_stream_t* GraphMap::execResultQuery(neo4j_connection_t* conn,
												 const std::string& query,
												 const neo4j_value_t& params) {
	neo4j_result_stream_t* results = neo4j_run(conn, query.c_str(), params);

	if(neo4j_check_failure(results)) {
		const char* msg = neo4j_error_message(results);
		throw std::runtime_error(msg);
	}

	return results;
}

std::string GraphMap::color2Str(const cv::Scalar& c) {
	cv::Mat3f bgrMat(cv::Vec3f(c[0], c[1], c[2]));
	cv::Mat3f hsvMat;
	cv::cvtColor(bgrMat, hsvMat, cv::COLOR_BGR2HSV);
	cv::Vec3f hsv = hsvMat.at<cv::Vec3f>(0);
	float h = hsv[0];		// range [0, 360]
	float s = hsv[1] / 255;	// range [0, 100]
	float v = hsv[2] / 255;	// range [0, 100]

	if(s < 25) {			// any gray value
		if(v < 25) {
			return "black";
		} else if(v > 85) {
			return "white";
		}

		return "gray";
	}

	if((330 < h && h <= 360) || (0 < h && h <= 15)) { // red
		return "red";
	} else if(15 < h && h <= 45) { // orange
		return "orange";
	} else if(45 < h && h <= 75) { // yellow
		return "yellow";
	} else if(75 < h && h <= 165) { // green
		return "green";
	} else if(165 < h && h <= 195) { // turquoise
		return "turquoise";
	} else if(195 < h && h <= 270) { // blue
		return "blue";
	} else if(270 < h && h <= 285) { // purple
		return "purple";
	} else if(285 < h && h <= 330) { // pink
		return "pink";
	}

	return "INVALID";
}

std::string GraphMap::dir2Str(const Eigen::Vector3f& watchDir,
							  const Eigen::Vector3f& offset) {
	auto nOff = offset.normalized();
	auto nWatch = watchDir.normalized();

	float dotUp = nOff.dot(Eigen::Vector3f(0,0,1));
	float angUp = std::acos(dotUp) * 180 / M_PI;
	if(0 <= angUp && angUp < 30) {
		return "below";
	} else if(180-30 < angUp && angUp <= 180) {
		return "above";
	}

	float dotWatchDir = nOff.dot(nWatch);
	float angWatchDir = std::acos(dotWatchDir) * 180 / M_PI;
	if(0 <= angWatchDir && angWatchDir < 10) {
		return "in front of";
	} else if(180-10 < angWatchDir && angWatchDir <= 180) {
		return "behind";
	}

	auto right = nWatch.cross(Eigen::Vector3f(0, 0, 1));
	float dotRight = nOff.dot(right);
	float angRight = std::acos(dotRight) * 180 / M_PI;
	if(0 <= angRight && angRight < 90) {
		return "left of";
	} else if(90 <= angRight && angRight <= 180) {
		return "right of";
	}

	return "REL_NOT_CONVERTIBLE";
}

void GraphMap::insertColor(const GraphMap::Detection& d,
						   const std::string& attrType,
						   neo4j_connection_t* conn) {
	const unsigned int numEntries = 3;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid_d"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	entries[1] = neo4j_map_kentry(neo4j_string("attr_type"),
								  neo4j_ustring(attrType.c_str(),
												static_cast<unsigned int>(attrType.length())));
	std::string color = color2Str(d.color);
	entries[2] = neo4j_map_kentry(neo4j_string("col"),
								  neo4j_ustring(color.c_str(),
												static_cast<unsigned int>(color.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	execInsertQuery(conn,
					"MATCH (o:object) "
					"WHERE o.uuid = $uuid_d "
					"CREATE (a:attribute {name: $col, type: $attr_type}) "
					"CREATE (o)"
					"-[:IS]->"
					"(a)",
					params);
}

void GraphMap::insertState(const GraphMap::Detection& d,
						   const std::string& attrType,
						   neo4j_connection_t* conn) {
	const unsigned int numEntries = 3;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid_d"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	entries[1] = neo4j_map_kentry(neo4j_string("attr_type"),
								  neo4j_ustring(attrType.c_str(),
												static_cast<unsigned int>(attrType.length())));
	std::string state = color2Str(d.color);
	if(state == "black") {
		state = "off";
	} else {
		state = "on";
	}
	entries[2] = neo4j_map_kentry(neo4j_string("state"),
								  neo4j_ustring(state.c_str(),
												static_cast<unsigned int>(state.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	execInsertQuery(conn,
					"MATCH (o:object) "
					"WHERE o.uuid = $uuid_d "
					"CREATE (a:attribute {name: $state, type: $attr_type}) "
					"CREATE (o)"
					"-[:IS]->"
					"(a)",
					params);
}

void GraphMap::insertScreenSize(const GraphMap::Detection& d,
								const std::string& attrType,
								neo4j_connection_t* conn) {
	const unsigned int numEntries = 3;
	neo4j_map_entry_t entries[numEntries];

	std::string uuidStr = boost::uuids::to_string(d.uuid);
	entries[0] = neo4j_map_kentry(neo4j_string("uuid_d"),
								  neo4j_ustring(uuidStr.c_str(),
												static_cast<unsigned int>(uuidStr.length())));
	entries[1] = neo4j_map_kentry(neo4j_string("attr_type"),
								  neo4j_ustring(attrType.c_str(),
												static_cast<unsigned int>(attrType.length())));
	float distx = d.bboxMax.x - d.bboxMin.x;
	float disty = d.bboxMax.y - d.bboxMin.y;
	float distz = d.bboxMax.z - d.bboxMin.z;
	float maxXY = std::max(distx, disty);
	float diag = std::sqrt(maxXY * maxXY + distz * distz);
	float zoll = diag * 100 / 2.54;

	// assume some common sizes
	int roundZoll;
	if(zoll < 24) {
		roundZoll = std::round(zoll);
	} else if(zoll < 28) {
		roundZoll = 28;
	} else if(zoll < 32) {
		roundZoll = 32;
	} else if(zoll < 42) {
		roundZoll = 42;
	} else if(zoll < 50) {
		roundZoll = 50;
	} else if(zoll < 55) {
		roundZoll = 55;
	} else if(zoll < 65) {
		roundZoll = 65;
	} else if(zoll < 75) {
		roundZoll = 75;
	} else if(zoll < 85) {
		roundZoll = 85;
	} else {
		roundZoll = std::round(zoll);
	}
	std::string size = std::to_string(roundZoll) + "\"";
	entries[2] = neo4j_map_kentry(neo4j_string("size"),
								  neo4j_ustring(size.c_str(),
												static_cast<unsigned int>(size.length())));
	neo4j_value_t params = neo4j_map(entries, numEntries);
	execInsertQuery(conn,
					"MATCH (o:object) "
					"WHERE o.uuid = $uuid_d "
					"CREATE (a:attribute {name: $size, type: $attr_type}) "
					"CREATE (o)"
					"-[:IS]->"
					"(a)",
					params);
}

GraphMap::DBFunMap GraphMap::dbFunMap = {
	{"insertColor", insertColor},
	{"insertState", insertState},
	{"insertScreenSize", insertScreenSize}
};

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(scenedesc::GraphMap, mira::MicroUnit);
