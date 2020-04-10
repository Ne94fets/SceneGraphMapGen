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
#include <exception>
#include <iomanip>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "../../Kinect/include/KinectGrabber.h"

namespace tf = tensorflow;

using namespace mira;


class Vector2TensorBuffer : public tf::TensorBuffer {
public:
	Vector2TensorBuffer(void* data, std::size_t len) : tf::TensorBuffer(data), m_len(len) {}
	//returns how many bytes we have in our buffer
	std::size_t size() const override {return m_len;}
	//needed so TF knows this isn't a child of some other buffer
	TensorBuffer* root_buffer() override { return this; }
	// Not actually sure why we need this, but it lets TF know where the memory for this tensor came from
	void FillAllocationDescription(tensorflow::AllocationDescription* proto) const override {}
	// A value of false indicates this TensorBuffer does not own the underlying data
	bool OwnsMemory() const override { return false; }
private:
	std::size_t m_len;
};

namespace recognition {

///////////////////////////////////////////////////////////////////////////////

/**
 * Recognizes object in RGB images and locates them in 3D space
 */
class ObjectRecognition3d : public MicroUnit {
	MIRA_OBJECT(ObjectRecognition3d)
public:

	typedef kinect::KinectGrabber::RegistrationData	RegistrationData;
	typedef kinect::KinectGrabber::RGBImgType		RGBImgType;
	typedef kinect::KinectGrabber::DepthImgType		DepthImgType;

	struct Detection {
		cv::Rect box;
		int type;
		float confidence;
	};

public:

	ObjectRecognition3d();
	virtual ~ObjectRecognition3d();

	template<typename Reflector>
	void reflect(Reflector& r) {
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

	void onRegistrationData(ChannelRead<RegistrationData> data);
	void onNewRGBImage(ChannelRead<RGBImgType> image);
	void onNewDepthImage(ChannelRead<DepthImgType> image);

	cv::Point3f getXYZ(int r, int c, float depth);

	// void onPoseChanged(ChannelRead<Pose2> pose);

	// void setPose(const Pose2& pose);

private:
	Channel<RGBImgType>		m_channelRGBMarked;
	tf::Session* m_session;
	tf::GraphDef m_graphDef;

	RegistrationData	m_regData;
	bool				m_hasRegData = false;
	DepthImgType		m_lastDepthImg;

	static std::map<int, std::string> m_lookupMap;
};

///////////////////////////////////////////////////////////////////////////////

ObjectRecognition3d::ObjectRecognition3d() {
	// TODO: further initialization of members, etc.{
}

ObjectRecognition3d::~ObjectRecognition3d() {
	tf::Status status = m_session->Close();
}

void ObjectRecognition3d::initialize() {
	tf::Status status = tf::NewSession(tf::SessionOptions(), &m_session);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	status = tf::ReadBinaryProto(tf::Env::Default(), "frozen_inference_graph.pb", &m_graphDef);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	// Add the graph to the session
	status = m_session->Create(m_graphDef);
	if(!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	subscribe<RegistrationData>("KinectRegData", &ObjectRecognition3d::onRegistrationData);
	subscribe<RGBImgType>("RGBImage", &ObjectRecognition3d::onNewRGBImage);
	subscribe<DepthImgType>("DepthImage", &ObjectRecognition3d::onNewDepthImage);

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &ObjectRecognition3d::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
	m_channelRGBMarked = publish<RGBImgType>("RGBImageMarked");
}

void ObjectRecognition3d::onRegistrationData(ChannelRead<ObjectRecognition3d::RegistrationData> data) {
	if(m_hasRegData)
		return;

	m_regData = *data;
	m_hasRegData = true;
}

void ObjectRecognition3d::onNewRGBImage(ChannelRead<ObjectRecognition3d::RGBImgType> image) {
	std::cout << "New RGB image" << std::endl;
	auto imgSize = image->size();

	tf::Tensor inTensor(tf::DataType::DT_UINT8,
						tf::TensorShape({1, imgSize.height(), imgSize.width(), 3}),
						new Vector2TensorBuffer(
							const_cast<void*>(reinterpret_cast<const void*>(image->data())),
							static_cast<size_t>(imgSize.width() * imgSize.height()) *
							3 * sizeof(uint8_t)));
	std::vector<std::pair<std::string, tf::Tensor>> inputs = {
		{"image_tensor", inTensor},
	};

	std::vector<tf::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	tf::Status status = m_session->Run(
	inputs, {
		"num_detections",
		"detection_boxes",
		"detection_scores",
		"detection_classes"
	},
	{},
	&outputs);
	if(!status.ok()) {
		std::cout << status.ToString() << "\n";
	}

	if(outputs.empty())
		return;

	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)
	std::cout << outputs[0].DebugString() << "\n";
//	std::cout << m_outputs[0].scalar<int32>()() << "\n";
	std::cout << outputs[1].DebugString() << "\n";
//	std::cout << m_outputs[1].scalar<int32>()() << "\n";
	std::cout << outputs[2].DebugString() << "\n";
//	std::cout << m_outputs[2].scalar<float>()() << "\n";
	std::cout << outputs[3].DebugString() << "\n";
//	std::cout << m_outputs[3].scalar<int64>()() << "\n";
	int32_t numDetections = static_cast<int32_t>(outputs[0].flat<float>().data()[0]);
	cv::Mat tmp = *image;
	cv::Mat outRGB(tmp);
	std::vector<Detection> detections;
	for(int32_t i = 0; i < numDetections; ++i) {
		Detection d;
		float ymin = outputs[1].flat<float>().data()[i * 4 + 0];
		float xmin = outputs[1].flat<float>().data()[i * 4 + 1];
		float ymax = outputs[1].flat<float>().data()[i * 4 + 2];
		float xmax = outputs[1].flat<float>().data()[i * 4 + 3];
		cv::Point2i
		p1(static_cast<int>(xmin * outRGB.size().width), static_cast<int>(ymin * outRGB.size().height)),
		p2(static_cast<int>(xmax * outRGB.size().width), static_cast<int>(ymax * outRGB.size().height));
		d.box = cv::Rect(p1, p2);
		cv::rectangle(outRGB, d.box, cv::Scalar(0, 0, 255), 4);

		d.confidence = outputs[2].flat<float>().data()[i];
		int type = static_cast<int>(outputs[3].flat<float>().data()[i]);
		d.type = type;

		std::stringstream topText;
		topText << m_lookupMap[type] << ": " << std::setprecision(0) << std::fixed << d.confidence * 100;
		int baseline = 0;
		double fontScale = 1;
		int thickness = 2;
		auto textSize = cv::getTextSize(topText.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
		auto textPos = p1;
		cv::rectangle(outRGB, cv::Rect(textPos + cv::Point2i(0, -textSize.height), textSize), cv::Scalar(0, 0, 255), -1);
		cv::putText(outRGB, topText.str(), textPos,
					cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
		textPos.y += textSize.height+4;

		// get array of depths for region of intererst.
		cv::Mat roi = m_lastDepthImg(d.box);
		std::vector<cv::Point3i> roiPoints;
		for(int y = d.box.y; y < d.box.y + d.box.height; ++y) {
			for(int x = d.box.x; x < d.box.x + d.box.width; ++x) {
				uint8_t depth = m_lastDepthImg(x, y);
				roiPoints.push_back(cv::Point3i(x, y, depth));
			}
		}

		// get mean of depths between 0.25 and 0.75 quantile
		auto const q1 = roiPoints.size() / 4;
		auto const q2 = roiPoints.size() / 2;
		auto const q3 = q1 + q2;
		std::sort(roiPoints.begin(), roiPoints.end(), [](const cv::Point3i & lhs, const cv::Point3i & rhs) {
			return lhs.z < rhs.z;
		});
		// average points in image space not in world space losing cone of camera intrinsics?
		cv::Point3i center = std::accumulate(&roiPoints[q1], &roiPoints[q3], cv::Point3i(0, 0, 0)) / float(q3 - q1);
		cv::Point3f centerRelativeToCam = getXYZ(center.y, center.x, center.z);
		std::stringstream textRelPos;
		textRelPos << center;
		cv::putText(outRGB, textRelPos.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255), thickness);
		detections.push_back(d);
	}
	RGBImgType rgbImgOut;
	outRGB.copyTo(rgbImgOut);
	auto wRGBMarked = m_channelRGBMarked.write();
	wRGBMarked->value() = rgbImgOut;
}

void ObjectRecognition3d::onNewDepthImage(ChannelRead<ObjectRecognition3d::DepthImgType> image) {
	m_lastDepthImg = *image;
}

cv::Point3f ObjectRecognition3d::getXYZ(int r, int c, float depth) {
	const float bad_point = std::numeric_limits<float>::quiet_NaN();
	if(!m_hasRegData)
		return cv::Point3f(bad_point, bad_point, bad_point);

	const float cx(m_regData.depth_p.cx), cy(m_regData.depth_p.cy);
	const float fx(1 / m_regData.depth_p.fx), fy(1 / m_regData.depth_p.fy);
	const float depth_val = depth / 1000.0f; //scaling factor, so that value of 1 is one meter.

	if(isnan(depth_val) || depth_val <= 0.001f) {
		//depth value is not valid
		return cv::Point3f(bad_point, bad_point, bad_point);
	} else {
		float x = (c + 0.5f - cx) * fx * depth_val;
		float y = (r + 0.5f - cy) * fy * depth_val;
		float z = depth_val;
		return cv::Point3f(x,y,z);
	}
}

std::map<int, std::string> ObjectRecognition3d::m_lookupMap = {
	{1,	"person"},
	{2,	"bicycle"},
	{3,	"car"},
	{4,	"motorcycle"},
	{5,	"airplane"},
	{6,	"bus"},
	{7,	"train"},
	{8,	"truck"},
	{9,	"boat"},
	{10,	"traffic light"},
	{11,	"fire hydrant"},
	{12,	"street sign"},
	{13,	"stop sign"},
	{14,	"parking meter"},
	{15,	"bench"},
	{16,	"bird"},
	{17,	"cat"},
	{18,	"dog"},
	{19,	"horse"},
	{20,	"sheep"},
	{21,	"cow"},
	{22,	"elephant"},
	{23,	"bear"},
	{24,	"zebra"},
	{25,	"giraffe"},
	{26,	"hat"},
	{27,	"backpack"},
	{28,	"umbrella"},
	{29,	"shoe"},
	{30,	"eye glasses"},
	{31,	"handbag"},
	{32,	"tie"},
	{33,	"suitcase"},
	{34,	"frisbee"},
	{35,	"skis"},
	{36,	"snowboard"},
	{37,	"sports ball"},
	{38,	"kite"},
	{39,	"baseball bat"},
	{40,	"baseball glove"},
	{41,	"skateboard"},
	{42,	"surfboard"},
	{43,	"tennis racket"},
	{44,	"bottle"},
	{45,	"plate"},
	{46,	"wine glass"},
	{47,	"cup"},
	{48,	"fork"},
	{49,	"knife"},
	{50,	"spoon"},
	{51,	"bowl"},
	{52,	"banana"},
	{53,	"apple"},
	{54,	"sandwich"},
	{55,	"orange"},
	{56,	"broccoli"},
	{57,	"carrot"},
	{58,	"hot dog"},
	{59,	"pizza"},
	{60,	"donut"},
	{61,	"cake"},
	{62,	"chair"},
	{63,	"couch"},
	{64,	"potted plant"},
	{65,	"bed"},
	{66,	"mirror"},
	{67,	"dining table"},
	{68,	"window"},
	{69,	"desk"},
	{70,	"toilet"},
	{71,	"door"},
	{72,	"tv"},
	{73,	"laptop"},
	{74,	"mouse"},
	{75,	"remote"},
	{76,	"keyboard"},
	{77,	"cell phone"},
	{78,	"microwave"},
	{79,	"oven"},
	{80,	"toaster"},
	{81,	"sink"},
	{82,	"refrigerator"},
	{83,	"blender"},
	{84,	"book"},
	{85,	"clock"},
	{86,	"vase"},
	{87,	"scissors"},
	{88,	"teddy bear"},
	{89,	"hair drier"},
	{90,	"toothbrush"},
	{91,	"hair brush"}
};


///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
