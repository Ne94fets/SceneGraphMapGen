#ifndef TRACKERGENERATOR_H
#define TRACKERGENERATOR_H

#include "RecognitionDataTypesExports.h"

#include <opencv2/tracking.hpp>

namespace recognitiondatatypes {

class MIRA_RECOGNITIONDATATYPES_EXPORT TrackerGenerator {
public:

	static cv::Ptr<cv::Tracker>	createKCFHOG();

	static void hogExtractor(const cv::Mat img, const cv::Rect roi, cv::Mat& feat);
};

} // namespace recognitiondatatypes

#endif // TRACKERGENERATOR_H
