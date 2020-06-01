#include "kinectdatatypes/Types.h"

namespace kinectdatatypes {

RegistrationData::RegistrationData() {

}

RegistrationData::RegistrationData(const libfreenect2::Freenect2Device::IrCameraParams& depth_p,
								   const libfreenect2::Freenect2Device::ColorCameraParams& rgb_p)
	: depth_p(depth_p),
	  rgb_p(rgb_p) {

}

} // namespace kinectdatatypes

#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>

#include <widgets/QtUtils.h>
#include <visualization/Visualization2D.h>
#include <visualization/ColormapProperty.h>

namespace visualisation {

/// A 2D visualization for different numbered image types.
template <typename Image>
class NumberedFrameVisualisationBase :  public Visualization2D
{
public:
	/** @name Constructor, destructor and reflect */
	//@{

	/// The constructor.
	NumberedFrameVisualisationBase() :
		mImageItem(NULL),
		mColormap("mira::GrayscaleColormap")
	{
		mImageChannel.setDataChangedCallback(
			boost::bind(&NumberedFrameVisualisationBase::dataChanged, this, _1));
		mImageChannel.setChannelChangedCallback(
			boost::bind(&NumberedFrameVisualisationBase::channelChanged, this));
		mMin =  0.0;
		mMax = -1.0;
		mAlpha = 1.0f;
		mImageSize = Size2i(0,0);
	}

	/// The destructor.
	virtual ~NumberedFrameVisualisationBase()
	{
		delete mImageItem;
	}

	/// The reflect method.
	template <typename Reflector>
	void reflect(Reflector& r)
	{
		Visualization2D::reflect(r);
		channelProperty(r, "Image", mImageChannel,
						"The channel with the image to display");

		r.roproperty("Size", mImageSize, "The size of the image");

		r.property("Colormap", mColormap, "The color palette",
				   ColormapProperty("mira::GrayscaleColormap"));
		r.property("Min", mMin,
				   setter(&NumberedFrameVisualisationBase::setMin, this),
				   "Min. value in the image (only valid for float and uint16 images)",
				   0.0f);
		r.property("Max", mMax,
				   setter(&NumberedFrameVisualisationBase::setMax, this),
				   "Max. value in the image (only valid for float and uint16 images)",
				   -1.0f);
		r.property("Alpha", mAlpha, "Opacity 0.0-1.0, default is 1.0", 1.0f);
	}

	//@}

public:
	/** @name Public implementation of Visualization2D */
	//@{

	virtual void setupScene(IVisualization2DSite* site)
	{
		QGraphicsScene* mgr = site->getSceneManager();
		mImageItem = new QGraphicsPixmapItem();
		mImageItem->setOpacity(mAlpha);
		mImageItem->scale(1.0, -1.0); // here we care about facing the correct way.
		mgr->addItem(mImageItem);
	}

	virtual QGraphicsItem* getItem()
	{
		return mImageItem;
	}

	virtual void setEnabled(bool enabled)
	{
		Visualization2D::setEnabled(enabled);
		mImageItem->setVisible(enabled);
	}

	//@}

public:
	/** @name Overriden methods of Visualization */
	//@{

	virtual DataConnection getDataConnection()
	{
		return DataConnection(mImageChannel);
	}

	//@}

private:
	void dataChanged(ChannelRead<Image> img)
	{
		const Img<>& untypedImg = img->value();
		QImage qimg = QtUtils::toQImage(untypedImg, mMin, mMax);
		if(qimg.isNull()) {
			error("Type", "Unsupported image format");
			return;
		}

		if(qimg.format()==QImage::Format_Indexed8) {
			if(mColormap.isValid())
				qimg.setColorTable(mColormap.getColorTable());
		}
		mImageSize = img->value().size();
		mImageItem->setPixmap(QPixmap::fromImage(qimg));
		mImageItem->setOpacity(mAlpha);
		ok("Type");
	}

	void channelChanged()
	{
	}

	void setMin(float min)
	{
		if (min < mMax)
			mMin = min;
	}

	void setMax(float max)
	{
		if (max > mMin)
			mMax = max;
	}

private:
	ChannelProperty<Image> mImageChannel;

	QGraphicsPixmapItem* mImageItem;
	Size2i mImageSize;

	ColormapProperty mColormap;
	float mMin, mMax;
	float mAlpha;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace visualisation

using namespace kinectdatatypes;

#define CREATE_IMG_VISUALIZATION(type,channels,name)			\
namespace visualisation {										\
class NumberedFrameVisualisation_##type##channels :                      \
	public NumberedFrameVisualisationBase<NumberedFrame<Img<type,channels>>>            \
{																\
	MIRA_META_OBJECT(NumberedFrameVisualisation_##type##channels,        \
		("Category", "Images")									\
		("Name", name)											\
		("Description", "Displays images"))						\
};																\
}																\
MIRA_CLASS_SERIALIZATION(visualisation::NumberedFrameVisualisation_##type##channels, mira::Visualization2D);

CREATE_IMG_VISUALIZATION(void,  1,"NumberedFrame<Img<>>")
CREATE_IMG_VISUALIZATION(uint8, 1,"NumberedFrame<Img<uint8,1>>")
CREATE_IMG_VISUALIZATION(uint8, 3,"NumberedFrame<Img<uint8,3>>")
CREATE_IMG_VISUALIZATION(uint16,1,"NumberedFrame<Img<uint16,1>>")
CREATE_IMG_VISUALIZATION(float, 1,"NumberedFrame<Img<float,1>>")
CREATE_IMG_VISUALIZATION(double,1,"NumberedFrame<Img<double,1>>")

