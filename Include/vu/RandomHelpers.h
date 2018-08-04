#pragma once

#include <map>
#include <random>
#include <vector>

#include <NDT/Tensor.h>

namespace vu {

static inline std::mt19937 & Generator() {
    static std::random_device randomDev;
    static std::mt19937 gen(randomDev());
    return gen;
}

//static inline int UniformIntSample(const int a, const int b) {
//
//    std::uniform_int_distribution<> distribution(a,b);
//
//    return distribution(Generator());
//
//}

template <typename Scalar>
static inline
typename std::enable_if<std::is_integral<Scalar>::value, Scalar>::type
UniformSample(const Scalar min, const Scalar max) {
    std::uniform_int_distribution<Scalar> distribution(min, max);
    return distribution(Generator());
}

template <typename Scalar>
static inline
typename std::enable_if<std::is_floating_point<Scalar>::value, Scalar>::type
UniformSample(const Scalar min = Scalar(0), const Scalar max = Scalar(1)) {
    std::uniform_real_distribution<Scalar> distribution(min, max);
    return distribution(Generator());
//    return min + UniformUnitSample()*(max-min);
}

template <typename T>
static inline const T & SelectFromUniformlyAtRandom(const std::vector<T> & vec, uint & index) {

    index = UniformIntSample(0,vec.size()-1);

    return vec[index];

}

template <typename T>
static inline const T & SelectFromUniformlyAtRandom(const std::vector<T> & vec) {

    uint index;

    return SelectFromUniformlyAtRandom(vec,index);

}

template <typename T>
static inline T RemoveFromUniformlyAtRandom(std::vector<T> & vec) {

    typename std::vector<T>::iterator it = vec.begin();

    std::advance(it, UniformIntSample(0,vec.size()-1));

    const T retVal = *it;

    vec.erase(it);

    return retVal;

}

template <typename T1, typename T2>
static inline std::pair<T1,T2> SelectFromUniformlyAtRandom(const std::map<T1,T2> & map) {

    typename std::map<T1,T2>::const_iterator it = map.begin();

    std::advance(it, UniformIntSample(0,map.size()-1));

    return *it;

}

template <typename T1, typename T2>
static inline std::pair<T1,T2> RemoveFromUniformlyAtRandom(const std::map<T1,T2> & map) {

    typename std::map<T1,T2>::const_iterator it = map.begin();

    std::advance(it, UniformIntSample(0,map.size()-1));

    const std::pair<T1,T2> retVal = *it;

    map.erase(it);

    return retVal;

}

//static inline float UniformUnitSample() {
//
//    return rand() / (float) RAND_MAX;
//
//}



template <typename Scalar>
static inline uint SelectIndexFromCumulativeDistribution(const std::vector<Scalar> & cdf) {

    Scalar r = UniformSample<Scalar>();

    return std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();

}

//[[deprecated("new version available")]]
template <typename T>
void FisherYatesShuffle(std::vector<T> & vec, const int N) {

    static std::random_device randomDev;
    static std::mt19937 gen(randomDev());

    typename std::vector<T>::iterator it = vec.begin();
    for (int distanceToEnd = vec.size(); distanceToEnd > (vec.size() - N); --distanceToEnd) {
        std::uniform_int_distribution<> distro(0,distanceToEnd-1);
        typename std::vector<T>::iterator it2 = it;
        std::advance(it2, distro(gen) );
        std::swap(*it,*it2);
        ++it;
    }

}


} // namespace vu