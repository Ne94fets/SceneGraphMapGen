#ifndef RGBDQUEUE_H
#define RGBDQUEUE_H

#include "KinectDataTypesExports.h"


#include <optional>
#include <mutex>
#include <queue>

#include <fw/ChannelReadWrite.h>
#include <image/Img.h>

using namespace mira;

namespace kinectdatatypes {

template<typename TChannel0, typename TChannel1>
class MIRA_KINECTDATATYPES_EXPORT RGBDQueue {
public:

	typedef std::pair<TChannel0, TChannel1>	ChannelPair;

public:
	RGBDQueue() {}

	void push0(const TChannel0& img) {
		std::lock_guard<std::mutex> guard(m_channel0Mutex);
		m_channel0Queue.push(img);
	}
	void push0(TChannel0&& img) {
		std::lock_guard<std::mutex> guard(m_channel0Mutex);
		m_channel0Queue.push(img);
	}

	void push1(const TChannel1& img) {
		std::lock_guard<std::mutex> guard(m_channel1Mutex);
		m_channel1Queue.push(img);
	}
	void push1(TChannel1&& img) {
		std::lock_guard<std::mutex> guard(m_channel1Mutex);
		m_channel1Queue.push(img);
	}

	std::optional<ChannelPair> getNewestSyncedPair()  {
		std::optional<ChannelPair> pair;

		// get frame number of front image
		std::lock_guard guard0(m_channel0Mutex);
		std::lock_guard guard1(m_channel1Mutex);
		if(m_channel1Queue.empty() || m_channel0Queue.empty()) {
			return pair;
		}

		bool skipping = false;
		unsigned int prevSequenceID = 0;

		while(!m_channel0Queue.empty() && !m_channel1Queue.empty()) {
			// pop rgb images before next depth image
			while(!m_channel0Queue.empty() && m_channel0Queue.front().sequenceID < m_channel1Queue.front().sequenceID) {
				std::cout << "No matching depth image. Dropping RGB image. FrameNumber: " << m_channel0Queue.front().sequenceID << std::endl;
				m_channel0Queue.pop();
			}

			if(m_channel0Queue.empty()) {
				break;
			}

			// pop depth images before next rgb image
			while(!m_channel1Queue.empty() && m_channel1Queue.front().sequenceID < m_channel0Queue.front().sequenceID) {
				std::cout << "No matching rgb image. Dropping depth image. FrameNumber: " << m_channel1Queue.front().sequenceID << std::endl;
				m_channel1Queue.pop();
			}

			if(m_channel1Queue.empty()) {
				break;
			}

			if(m_channel0Queue.front().sequenceID != m_channel1Queue.front().sequenceID) {
				continue;
			}

			pair = std::make_pair(m_channel0Queue.front(), m_channel1Queue.front());

			// rgb and depth image should be synchronized and have same frame number now
			assert(pair->first.sequenceID == pair->second.sequenceID);

			m_channel0Queue.pop();
			m_channel1Queue.pop();
			if(skipping) {
				std::cout << "Skipping frames: " << prevSequenceID << std::endl;
			}
			skipping = true;
			prevSequenceID = pair->first.sequenceID;
		}

		return pair;
	}

private:
	std::mutex	m_channel0Mutex;
	std::mutex	m_channel1Mutex;

	std::queue<TChannel0>	m_channel0Queue;
	std::queue<TChannel1>	m_channel1Queue;
};

} // namespace kinectdatatypes

#endif // RGBDQUEUE_H
