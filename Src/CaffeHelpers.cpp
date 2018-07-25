/***
 * © Tanner Schmidt 2018
 */

#ifdef HAS_CAFFE

#include <vu/CaffeHelpers.h>
#include <vu/FileHelpers.h>

#include <regex>

namespace vu {

void InitCaffe(char * executableName) {

    char * arg = executableName; //buffer.data();
    char * * argv = &arg;
    int argc = 1;
    caffe::GlobalInit(&argc,&argv);

}

void CaffeSwizzle(const NDT::ConstVolume<unsigned char> & source,
                  NDT::Volume<float> & destination) {

    for (int c = 0; c < destination.DimensionSize(2); ++c) {
        for (int y = 0; y < destination.DimensionSize(1); ++y) {
            for (int x = 0; x < destination.DimensionSize(0); ++x) {
                destination(x,y,c) = source(c,x,y) / 255.f;
            }
        }
    }

}

int GetLatestSnapshot(const std::string & experimentDir) {

    int latestSnap = 0;
    std::vector<std::string> experimentDirContents = GetDirectoryContents(experimentDir);
    std::regex snapRegex("snap_iter_(\\d+).caffemodel");
    for (std::string & content : experimentDirContents) {
        std::smatch match;
        if (std::regex_search(content,match,snapRegex)) {
            latestSnap = std::max(latestSnap,atoi(match.str(1).c_str()));
        }
    }
    return latestSnap;

}

template <typename T>
void MakeDeconvolutionalWeightsBilinear(caffe::Net<T> & net,
                                        const std::string layerName) {

    boost::shared_ptr<caffe::Layer<T> > layer = net.layer_by_name(layerName);

    if (!layer) {
        throw std::runtime_error("no layer " + layerName);
    }

    if (strcmp(layer->type(),"Deconvolution")) {
        throw std::runtime_error(layerName + " is not a deconvolutional layer");
    }

    if (layer->blobs().size() != 2) {
        throw std::runtime_error("expected 2 blobs, found " + std::to_string(layer->blobs().size()));
    }

    boost::shared_ptr<caffe::Blob<T> > weightBlob = layer->blobs()[0];
    boost::shared_ptr<caffe::Blob<T> > biasBlob = layer->blobs()[1];

    const uint channelsIn = weightBlob->shape()[1];
    const uint channelsOut = weightBlob->shape()[0];

    std::cout << channelsIn << " channels in, " << channelsOut << " channels out" << std::endl;

    if (channelsIn != channelsOut) {
        throw std::runtime_error("channels in does not match channels out for layer " + layerName);
    }

    // wipe previous weights and biases
    caffe::caffe_set(weightBlob->count(),0.f,weightBlob->mutable_cpu_data());
    caffe::caffe_set(biasBlob->count(),0.f,biasBlob->mutable_cpu_data());

    const uint widthOut = weightBlob->shape()[3];
    const uint heightOut = weightBlob->shape()[2];

    const uint stride = layer->layer_param().convolution_param().stride().Get(0);

    // put in bilinear weights
    for (uint c = 0; c < channelsOut; ++c) {

        for (uint h = 0; h < heightOut; ++h) {

            const T hInterp = 1 - ((std::fabs((int)h - (int)(heightOut/2) + 0.5) ) / stride);

            for (uint w = 0; w < widthOut; ++w) {

                const T wInterp = 1 - ((std::fabs((int)w - (int)(widthOut/2) + 0.5) ) / stride);

                weightBlob->mutable_cpu_data()[weightBlob->offset(c,c,h,w)] = wInterp*hInterp;

//                if (c == 0) {
//                    std::cout << weightBlob->mutable_cpu_data()[weightBlob->offset(c,c,h,w)] << "  ";
//                }

            }

//            if (c == 0) {
//                std::cout << std::endl;
//            }

        }

    }

}

template void MakeDeconvolutionalWeightsBilinear(caffe::Net<float> & net,
                                                 const std::string layerName);

} // namespace vu

#endif // HAS_CAFFE