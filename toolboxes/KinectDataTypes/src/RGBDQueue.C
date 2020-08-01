#include "kinectdatatypes/RGBDQueue.h"

namespace kinectdatatypes {

RGBDQueue::RGBDQueue() {

}

void RGBDQueue::push(ChannelRead<RGBImgType> img) {
	std::lock_guard<std::mutex> rgbMutex(m_rgbMutex);
	m_rgbQueue.push(img);
}

void RGBDQueue::push(ChannelRead<DepthImgType> img) {
	std::lock_guard<std::mutex> depthMutex(m_depthMutex);
	m_depthQueue.push(img);
}

std::optional<RGBDQueue::ChannelReadPair> RGBDQueue::getNewestSyncedPair() {
	std::optional<ChannelReadPair> pair;

	// get frame number of front image
	std::lock_guard rgbGuard(m_rgbMutex);
	std::lock_guard depthGuard(m_depthMutex);
	if(m_depthQueue.empty() || m_rgbQueue.empty()) {
		return {};
	}

	bool skipping = false;
	unsigned int prevSequenceID = 0;

	while(m_rgbQueue.size() > 1 && m_depthQueue.size() > 1) {
		// pop rgb images before next depth image
		while(!m_rgbQueue.empty() && m_rgbQueue.front()->sequenceID < m_depthQueue.front()->sequenceID) {
			std::cout << "No matching depth image. Dropping RGB image. FrameNumber: " << m_rgbQueue.front()->sequenceID << std::endl;
			m_rgbQueue.pop();
		}

		if(m_rgbQueue.empty()) {
			break;
		}

		// pop depth images before next rgb image
		while(!m_depthQueue.empty() && m_depthQueue.front()->sequenceID < m_rgbQueue.front()->sequenceID) {
			std::cout << "No matching rgb image. Dropping depth image. FrameNumber: " << m_depthQueue.front()->sequenceID << std::endl;
			m_depthQueue.pop();
		}

		if(m_depthQueue.empty()) {
			break;
		}

		if(m_rgbQueue.front()->sequenceID != m_depthQueue.front()->sequenceID) {
			continue;
		}

		pair = std::make_pair(m_rgbQueue.front(), m_depthQueue.front());

		// rgb and depth image should be synchronized and have same frame number now
		assert(pair->first->sequenceID == pair->second->sequenceID);

		m_rgbQueue.pop();
		m_depthQueue.pop();
		if(skipping) {
			std::cout << "Skipping frames: " << prevSequenceID << std::endl;
		}
		skipping = true;
		prevSequenceID = pair->first->sequenceID;
	}

	return pair;
}

} // namespace kinectdatatypes
