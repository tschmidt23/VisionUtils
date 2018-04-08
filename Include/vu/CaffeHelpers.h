#include <caffe/caffe.hpp>

#include <string>

namespace vu {

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