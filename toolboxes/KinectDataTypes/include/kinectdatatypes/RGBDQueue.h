#ifndef RGBDQUEUE_H
#define RGBDQUEUE_H

#include "KinectDataTypesExports.h"


#include <optional>
#include <mutex>
#include <condition_variable>
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
		std::unique_lock<std::mutex> lock(m_mutex);
		m_channel0Queue.push(img);
		dropTilID(m_channel1Queue, m_channel0Queue.front().sequenceID);
		checkForPair();
		lock.unlock();
		m_condition.notify_one();
	}
	void push0(TChannel0&& img) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_channel0Queue.push(img);
		dropTilID(m_channel1Queue, m_channel0Queue.front().sequenceID);
		checkForPair();
		lock.unlock();
		m_condition.notify_one();
	}

	void push1(const TChannel1& img) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_channel1Queue.push(img);
		dropTilID(m_channel0Queue, m_channel1Queue.front().sequenceID);
		checkForPair();
		lock.unlock();
		m_condition.notify_one();
	}
	void push1(TChannel1&& img) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_channel1Queue.push(img);
		dropTilID(m_channel0Queue, m_channel1Queue.front().sequenceID);
		checkForPair();
		lock.unlock();
		m_condition.notify_one();
	}

	ChannelPair getNewestSyncedPair()  {
		std::unique_lock lock(m_mutex);

		m_condition.wait(lock, [this]{ return m_currentPair.has_value(); });

		auto pair = m_currentPair;
		m_currentPair = {};

		return *pair;
	}

	template<typename TChannel>
	static void dropTilID(std::queue<TChannel>& queue, size_t id) {
		while(!queue.empty() && queue.front().sequenceID < id) {
			std::cout << "No matching image. Dropping image. FrameNumber: " << queue.front().sequenceID << std::endl;
			queue.pop();
		}
	}

private:
	std::condition_variable	m_condition;
	std::mutex				m_mutex;

	std::queue<TChannel0>	m_channel0Queue;
	std::queue<TChannel1>	m_channel1Queue;

	std::optional<ChannelPair>	m_currentPair;

private:
	void checkForPair() {
		if(m_channel0Queue.empty() || m_channel1Queue.empty() ||
				m_channel0Queue.front().sequenceID != m_channel1Queue.front().sequenceID) {
			return;
		}

		if(m_currentPair) {
			std::cout << "Skipping pair. FrameNumber: " << m_currentPair->first.sequenceID << std::endl;
		}

		m_currentPair = std::make_pair(m_channel0Queue.front(), m_channel1Queue.front());
		m_channel0Queue.pop();
		m_channel1Queue.pop();
	}
};

} // namespace kinectdatatypes

#endif // RGBDQUEUE_H
