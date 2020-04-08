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

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

namespace tf = tensorflow;

using namespace mira;


class Vector2TensorBuffer : public tf::TensorBuffer
{
public:
	Vector2TensorBuffer(void* data, std::size_t len) : tf::TensorBuffer(data), m_len(len) {}
	//returns how many bytes we have in our buffer
	std::size_t size() const override {return m_len;};
	//needed so TF knows this isn't a child of some other buffer
	TensorBuffer* root_buffer() override { return this; }
	// Not actually sure why we need this, but it lets TF know where the memory for this tensor came from
	void FillAllocationDescription(tensorflow::AllocationDescription* proto) const override{};
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
class ObjectRecognition3d : public MicroUnit
{
MIRA_OBJECT(ObjectRecognition3d)

public:

	ObjectRecognition3d();
	virtual ~ObjectRecognition3d();

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
	//Channel<Img<>> mChannel;
	tf::Session* m_session;
	tf::GraphDef m_graphDef;

	std::vector<tf::Tensor> m_outputs;
};

///////////////////////////////////////////////////////////////////////////////

ObjectRecognition3d::ObjectRecognition3d()
{
	// TODO: further initialization of members, etc.
}

ObjectRecognition3d::~ObjectRecognition3d()
{
	tf::Status status = m_session->Close();
}

void ObjectRecognition3d::initialize()
{
	tf::Status status = NewSession(tf::SessionOptions(), &m_session);
	if (!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	status = ReadBinaryProto(tf::Env::Default(), "models/graph.pb", &m_graphDef);
	if (!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	// Add the graph to the session
	status = m_session->Create(m_graphDef);
	if (!status.ok()) {
		throw std::runtime_error(status.ToString());
	}

	subscribe<RGBImgType>("RGBImage", &ObjectRecognition3d::onNewRGBImage);
	subscribe<DepthImgType>("DepthImage", &ObjectRecognition3d::onNewDepthImage);

	// TODO: subscribe and publish all required channels
	//subscribe<Pose2>("Pose", &ObjectRecognition3d::onPoseChanged);
	//mChannel = publish<Img<>>("Image");
}

void ObjectRecognition3d::onNewRGBImage(ChannelRead<ObjectRecognition3d::RGBImgType> image)
{
	std::vector<uint8_t> tmp(image->data(), image->data() + image->size().size()*3);
	auto imgSize = image->size();
	Vector2TensorBuffer v2B((void*)image->data(), imgSize.width() * imgSize.height());
	tf::Tensor inTensor(tf::DataType::DT_INT8, tf::TensorShape(), &v2B);
	std::vector<std::pair<std::string, tf::Tensor>> inputs = {
		{"image_tensor", inTensor},
	};

	// Run the session, evaluating our "c" operation from the graph
	tf::Status status = m_session->Run(inputs, {"detection_boxes", "detection_scores"}, {}, &m_outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
	}

	auto output_c = m_outputs[0].scalar<float>();
	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)
	std::cout << m_outputs[0].DebugString() << "\n"; // Tensor<type: float shape: []>
	std::cout << output_c() << "\n";
}

void ObjectRecognition3d::onNewDepthImage(ChannelRead<ObjectRecognition3d::DepthImgType> image)
{

}

///////////////////////////////////////////////////////////////////////////////

}

MIRA_CLASS_SERIALIZATION(recognition::ObjectRecognition3d, mira::MicroUnit);
