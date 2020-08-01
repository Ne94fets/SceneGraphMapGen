#ifndef RGBDQUEUE_H
#define RGBDQUEUE_H

#include "KinectDataTypesExports.h"


#include <optional>
#include <mutex>
#include <queue>

#include <fw/MicroUnit.h>
#include <image/Img.h>

using namespace mira;

namespace kinectdatatypes {

class MIRA_KINECTDATATYPES_EXPORT RGBDQueue {
public:

	typedef Img<uint8_t, 3>	RGBImgType;
	typedef Img<float, 1>	DepthImgType;

	typedef std::pair<ChannelRead<RGBImgType>, ChannelRead<DepthImgType>>	ChannelReadPair;

public:
	RGBDQueue();

	void push(ChannelRead<RGBImgType> img);
	void push(ChannelRead<DepthImgType> img);

	std::optional<ChannelReadPair> getNewestSyncedPair();

private:
	std::mutex	m_depthMutex;
	std::mutex	m_rgbMutex;
	std::queue<ChannelRead<DepthImgType>>	m_depthQueue;
	std::queue<ChannelRead<RGBImgType>>		m_rgbQueue;
};

} // namespace kinectdatatypes

#endif // RGBDQUEUE_H
