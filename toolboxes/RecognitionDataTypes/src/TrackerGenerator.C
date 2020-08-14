#include "recognitiondatatypes/TrackerGenerator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

namespace recognitiondatatypes {

cv::Ptr<cv::Tracker> TrackerGenerator::createKCFHOG() {
	// KCF on HOG tracker params
	cv::TrackerKCF::Params params;
	params.desc_pca = cv::TrackerKCF::GRAY | cv::TrackerKCF::CN;
	params.desc_npca = 0;
	params.compress_feature = true;
	params.compressed_size = 9;

	auto tracker = cv::TrackerKCF::create(params);
	tracker->setFeatureExtractor(hogExtractor);

	return tracker;
}

void TrackerGenerator::hogExtractor(const cv::Mat img, const cv::Rect roi, cv::Mat& feat) {
	cv::Mat sobel[2];
	cv::Mat patch;
	cv::Rect region=roi;

	// extract patch inside the image
	if(roi.x<0){region.x=0;region.width+=roi.x;}
	if(roi.y<0){region.y=0;region.height+=roi.y;}
	if(roi.x+roi.width>img.cols)region.width=img.cols-roi.x;
	if(roi.y+roi.height>img.rows)region.height=img.rows-roi.y;
	if(region.width>img.cols)region.width=img.cols;
	if(region.height>img.rows)region.height=img.rows;

	patch=img(region).clone();
	cv::cvtColor(patch,patch, CV_BGR2GRAY);

	// add some padding to compensate when the patch is outside image border
	int addTop,addBottom, addLeft, addRight;
	addTop=region.y-roi.y;
	addBottom=(roi.height+roi.y>img.rows?roi.height+roi.y-img.rows:0);
	addLeft=region.x-roi.x;
	addRight=(roi.width+roi.x>img.cols?roi.width+roi.x-img.cols:0);

	cv::copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,cv::BORDER_REPLICATE);

	int rows = patch.rows / 8;
	int cols = patch.cols / 8;

	cv::HOGDescriptor hog(cv::Size(rows * 8, cols * 8), cv::Size(16, 16), cv::Size(8,8),
						  cv::Size(8,8), 9);

	std::vector<float> desc;
	std::vector<cv::Point> loc;
	hog.compute(patch, desc, cv::Size(8,8), cv::Size(8,8), loc);

	feat.push_back(desc);
	feat = feat.reshape(9);

//	cv::Mat gx, gy;
//	cv::Sobel(patch, gx, CV_32F, 1, 0, 1);
//	cv::Sobel(patch, gy, CV_32F, 0, 1, 1);

//	cv::Mat mag, angle;
//	cv::cartToPolar(gx, gy, mag, angle, false);

//	std::vector<cv::Mat> hists(9, cv::Mat(patch.rows, patch.cols, CV_32F));
//	for(int y = 0; y < rows; ++y) {
//		for(int x = 0; x < cols; ++x) {
//			float sum = 0;
//			std::vector<float> hist(9, 0);
//			cv::Rect histRoi(x*8, y*8, 8, 8);
//			cv::Mat histMag = mag(histRoi);
//			cv::Mat histAng = angle(histRoi);
//			for(auto itMag = histMag.begin<float>(), itAngle = histAng.begin<float>();
//				itMag != histMag.end<float>(); ++itMag, ++itAngle) {
//				float m = *itMag;
//				float a = *itAngle;
//				float binf = a / (2*M_PI/9);
//				int bin = static_cast<int>(std::floor(binf));
//				hist.at(bin) += (1 - (binf - bin)) * m;
//				hist.at((bin + 1) % 9) += (binf - bin) * m;
//				sum += m;
//			}

//			for(size_t i = 0; i < hist.size(); ++i) {
//				hists[i].at<float>(y, x) = hist[i] / sum;
//			}
//		}
//	}

//	cv::merge(hists, feat);
}


} // namespace recognitiondatatypes
