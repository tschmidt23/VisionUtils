#pragma once

/***
 * © Tanner Schmidt 2018
 */


#include <caffe/caffe.hpp>

#include <NDT/Tensor.h>

#include <string>

namespace vu {

void InitCaffe(char * executableName);

void CaffeSwizzle(const NDT::ConstVolume<unsigned char> & source,
                  NDT::Volume<float> & destination);

int GetLatestSnapshot(const std::string & experimentDir);

template <template <typename> class LayerT, typename T>
inline boost::shared_ptr<LayerT<T> > GetLayer(const caffe::Net<T> & net,
                                              const std::string & layerName) {
    boost::shared_ptr<caffe::Layer<T> > layer = net.layer_by_name(layerName);
    if (layer) {
        boost::shared_ptr<LayerT<T> > layer_ = boost::static_pointer_cast<LayerT<T> >(layer);
        if (!layer_) {
            throw std::runtime_error("layer '" + layerName + "' does not seem to be of the requested type");
        }
        return layer_;
    }
    return boost::shared_ptr<LayerT<T> >(nullptr);
}

template <template <typename> class LayerT, typename T>
inline boost::shared_ptr <LayerT<T>> GetLayerOrDie(const caffe::Net <T> & net,
                                                   const std::string & layerName) {
    boost::shared_ptr <LayerT<T>> layer = GetLayer<LayerT, T>(net, layerName);
    if (!layer) {
        throw std::runtime_error("no layer named '" + layerName + "' found");
    }
    return layer;
}

template <typename T>
inline boost::shared_ptr <caffe::Blob<T>> GetBlobOrDie(const caffe::Net <T> & net,
                                                       const std::string & blobName) {

    boost::shared_ptr <caffe::Blob<T>> blob = net.blob_by_name(blobName);
    if (!blob) {
        throw std::runtime_error("no blob named '" + blobName + "' found");
    }
    return blob;

}

} // namespace vu