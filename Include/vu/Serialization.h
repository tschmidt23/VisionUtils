#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <map>
#include <vector>

#include <Eigen/Core>

#include <vu/DualQuaternion.h>

#include <NDT/Tensor.h>

#include <pangolin/utils/file_utils.h>

namespace vu {

namespace internal {

template <typename T, bool Fundamental> struct FallbackSerializer;


template <typename T>
struct DirectlySerializable {

    inline void Serialize(std::ostream & stream) const {

        stream.write(reinterpret_cast<const char *>(this), sizeof(T));

    }

    inline void Deserialize(std::istream & stream) {

        stream.read(reinterpret_cast<char *>(this), sizeof(T));

    }

};

/*
 * For fundamental types, the memory can be written directly to the stream
 */
template <typename T> struct FallbackSerializer<T, true> {

    static inline void Serialize(
            std::ostream & stream,
            const T val) {

        stream.write(reinterpret_cast<const char *>(&val),sizeof(T));

    }

    static inline void Deserialize(
            std::istream & stream,
            T & val) {

        stream.read(reinterpret_cast<char *>(&val),sizeof(T));

    }

};

/*
 * For all other non-specialized types, the serialization is delegated to the object itself.
 * This won't work for all objects (if they don't implement Serialize/Deserialize, but it
 * does allow for the serialization of user-defined types without modifying this file.
 */
template <typename T> struct FallbackSerializer<T,false> {

    static inline void Serialize(
            std::ostream & stream,
            const T val) {

        val.Serialize(stream);

    }

    static inline void Deserialize(
            std::istream & stream,
            T & val) {

        val.Deserialize(stream);

    }

};

/*
 * The default serializer provides fallback functionality for non-specialized types
 */
template <typename T>
struct Serializer {

    static inline void Serialize(
            std::ostream & stream,
            const T val) {

        FallbackSerializer<T, std::is_fundamental<T>::value>::Serialize(stream, val);

    }

    static inline void Deserialize(
            std::istream & stream,
            T & val) {

        FallbackSerializer<T, std::is_fundamental<T>::value>::Deserialize(stream, val);

    }

};

template <>
struct Serializer<std::string> {

    static void Serialize(std::ostream & stream,
                          const std::string & s) {

        const std::size_t size = s.size();
        stream.write(reinterpret_cast<const char *>(&size),sizeof(std::size_t));
        stream.write(s.c_str(),size);

    }

    static void Deserialize(std::istream & stream,
                            std::string & s) {

        std::size_t size;
        stream.read(reinterpret_cast<char *>(&size), sizeof(std::size_t));

        std::vector<char> buffer(size);
        stream.read(buffer.data(),size);

        s = std::string(buffer.data(),size);

    }

};

template <typename T>
struct Serializer<std::set<T> > {

    static void Serialize(std::ostream & stream,
                          const std::set<T> & s) {

        const std::size_t size = s.size();
        stream.write(reinterpret_cast<const char *>(&size),sizeof(std::size_t));

        for (const T & element : s) {
            Serializer<T>::Serialize(stream,element);
        }

    }

    static void Deserialize(std::istream & stream,
                            std::set<T> & s) {

        std::size_t size;
        stream.read(reinterpret_cast<char *>(&size),sizeof(std::size_t));

        for (std::size_t i = 0; i < size; ++i) {
            T element;
            Serializer<T>::Deserialize(stream,element);
            s.insert(element);
        }

    }

};

template <typename T, typename Alloc>
struct Serializer<std::vector<T, Alloc> > {

    static void Serialize(std::ostream & stream,
                          const std::vector<T, Alloc> & v) {

        const std::size_t size = v.size();
        stream.write(reinterpret_cast<const char *>(&size),sizeof(std::size_t));

        for (const T & element : v) {
            Serializer<T>::Serialize(stream,element);
        }

    }

    static void Deserialize(std::istream & stream,
                            std::vector<T, Alloc> & v) {

        std::size_t size;
        stream.read(reinterpret_cast<char *>(&size),sizeof(std::size_t));

        v.resize(size);
        for (T & element : v) {
            Serializer<T>::Deserialize(stream,element);
        }

    }

};

template <uint D, typename T>
struct Serializer<NDT::ManagedTensor<D,T,NDT::HostResident> > {

    static void Serialize(std::ostream & stream,
                          const NDT::ManagedTensor<D,T,NDT::HostResident> & image) {

        const Eigen::Matrix<uint,D,1> dimensions = image.Dimensions();
        stream.write(reinterpret_cast<const char *>(dimensions.data()), D * sizeof(uint));

        stream.write(reinterpret_cast<const char *>(image.Data()), image.SizeBytes());

    }

    static void Deserialize(std::istream & stream,
                            NDT::ManagedTensor<D,T,NDT::HostResident> & image) {

        Eigen::Matrix<uint,D,1> dimensions;
        stream.read(reinterpret_cast<char *>(dimensions.data()), D * sizeof(uint));

        image.Resize(dimensions);

        stream.read(reinterpret_cast<char *>(image.Data()), image.SizeBytes());

    }

};

template <typename T1, typename T2>
struct Serializer<std::map<T1,T2> > {

    static void Serialize(std::ostream & stream,
                          const std::map<T1,T2> & m) {

        const std::size_t size = m.size();
        stream.write(reinterpret_cast<const char *>(&size),sizeof(std::size_t));

        for (auto element : m) {
            Serializer<T1>::Serialize(stream,element.first);
            Serializer<T2>::Serialize(stream,element.second);
        }

    }

    static void Deserialize(std::istream & stream,
                            std::map<T1,T2> & m) {

        std::size_t size;
        stream.read(reinterpret_cast<char *>(&size),sizeof(std::size_t));

        for (std::size_t i = 0; i < size; ++i) {

            T1 key;
            T2 val;

            Serializer<T1>::Deserialize(stream,key);
            Serializer<T2>::Deserialize(stream,val);

            m[key] = val;

        }

    }

};

template <typename ... Ts>
struct IsFullyFundamental;

template <typename Head, typename ... Tails>
struct IsFullyFundamental<Head,Tails...> {

    static constexpr bool Value = std::is_fundamental<Head>::value && IsFullyFundamental<Tails...>::Value;

};

template <>
struct IsFullyFundamental<> {

    static constexpr bool Value = true;

};

template <int I, typename ... Ts>
struct TupleSerializer {

    static inline void Serialize(std::ostream & stream,
                                 const std::tuple<Ts...> & t) {

        const auto & val = std::get<sizeof...(Ts) - I>(t);

        Serializer<typename std::decay<decltype(val)>::type>::Serialize(stream, val);

        TupleSerializer<I-1,Ts...>::Serialize(stream, t);

    }

    static inline void Deserialize(std::istream & stream,
                                   std::tuple<Ts...> & t) {

        auto & val = std::get<sizeof...(Ts) - I>(t);

        Serializer<typename std::decay<decltype(val)>::type>::Deserialize(stream, val);

        TupleSerializer<I-1,Ts...>::Deserialize(stream, t);

    }

};

template <typename ... Ts>
struct TupleSerializer<0, Ts...> {

    static inline void Serialize(std::ostream & stream,
                                 const std::tuple<Ts...> & t) { }

    static inline void Deserialize(std::istream & stream,
                                   std::tuple<Ts...> & t) { }


};

template <typename ... Ts>
struct Serializer<std::tuple<Ts...> > {

    static void Serialize(std::ostream & stream,
                          const std::tuple<Ts...> & t) {

        if (IsFullyFundamental<Ts...>::Value) {

            stream.write(reinterpret_cast<const char *>(&t), sizeof(t));

        } else {

            TupleSerializer<sizeof...(Ts), Ts...>::Serialize(stream, t);

        }

    }

    static void Deserialize(std::istream & stream,
                            std::tuple<Ts...> & t) {

        if (IsFullyFundamental<Ts...>::Value) {

            stream.read(reinterpret_cast<char *>(&t), sizeof(t));

        } else {

            TupleSerializer<sizeof...(Ts),Ts...>::Deserialize(stream, t);

        }

    }

};

template <typename Scalar, int Cols, int Options>
struct Serializer<Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Options> > {

    static void Serialize(std::ostream & stream,
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Options> & m) {

        const Eigen::Index n = m.rows();
        stream.write(reinterpret_cast<const char *>(&n), sizeof(Eigen::Index));
        stream.write(reinterpret_cast<const char *>(m.data()), n * sizeof(Scalar));

    }

    static void Deserialize(std::istream & stream,
                            Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Options> & m) {

        Eigen::Index n;
        stream.read(reinterpret_cast<char *>(&n), sizeof(Eigen::Index));
        m = Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Options>(n, Cols);
        stream.read(reinterpret_cast<char *>(m.data()), n * sizeof(Scalar));

    }

};

template <typename Scalar, int Rows, int Cols, int Options>
struct Serializer<Eigen::Matrix<Scalar, Rows, Cols, Options> > {

    static void Serialize(std::ostream & stream,
                          const Eigen::Matrix<Scalar, Rows, Cols, Options> & m) {

        stream.write(reinterpret_cast<const char *>(m.data()), Rows * Cols * sizeof(Scalar));

    }

    static void Deserialize(std::istream & stream,
                            Eigen::Matrix<Scalar, Rows, Cols, Options> & m) {

        stream.read(reinterpret_cast<char *>(m.data()), Rows * Cols * sizeof(Scalar));

    }

};

template <typename Scalar, int D>
struct Serializer<Eigen::AlignedBox<Scalar, D> > {

    static inline void Serialize(std::ostream & stream,
                                 const Eigen::AlignedBox<Scalar, D> & b) {

        Serializer<Eigen::Matrix<Scalar,D,1> >::Serialize(stream, b.min());

        Serializer<Eigen::Matrix<Scalar,D,1> >::Serialize(stream, b.max());

    }

    static void Deserialize(std::istream & stream,
                            Eigen::AlignedBox<Scalar, D> & b) {

        Serializer<Eigen::Matrix<Scalar,D,1> >::Deserialize(stream, b.min());

        Serializer<Eigen::Matrix<Scalar,D,1> >::Deserialize(stream, b.max());

    }

};

template <typename Scalar, int Options>
struct Serializer<Eigen::Quaternion<Scalar, Options> > {

    static inline void Serialize(std::ostream & stream,
                                 const Eigen::Quaternion<Scalar, Options> & q) {

        Serializer<Eigen::Matrix<Scalar, 4, 1, Options> >::Serialize(stream, q.coeffs());

    }

    static inline void Deserialize(std::istream & stream,
                                   Eigen::Quaternion<Scalar, Options> & q) {

        Serializer<Eigen::Matrix<Scalar, 4, 1, Options> >::Deserialize(stream, q.coeffs());

    }

};

template <typename Scalar, int Options>
struct Serializer<DualQuaternion<Scalar, Options> > {

    static void Serialize(std::ostream & stream,
                          const DualQuaternion<Scalar, Options> & dq) {

        const Eigen::Quaternion<Scalar, Options> & nondual = dq.Nondual();

        Serializer<Eigen::Quaternion<Scalar, Options> >::Serialize(stream, nondual);

        const Eigen::Quaternion<Scalar, Options> & dual = dq.Dual();

        Serializer<Eigen::Quaternion<Scalar, Options> >::Serialize(stream, dual);

    }

    static void Deserialize(std::istream & stream,
                            DualQuaternion<Scalar, Options> & dq) {

        Eigen::Quaternion<Scalar, Options> nondual, dual;

        Serializer<Eigen::Quaternion<Scalar, Options> >::Deserialize(stream, nondual);

        Serializer<Eigen::Quaternion<Scalar, Options> >::Deserialize(stream, dual);

        dq = DualQuaternion<Scalar,Options>(nondual, dual);

    }

};

template <typename Scalar>
struct Serializer<Sophus::SE3<Scalar> > {

    static void Serialize(std::ostream & stream,
                          const Sophus::SE3<Scalar> & T) {

        Sophus::SO3<Scalar> R = T.so3();

        Eigen::Quaternion<Scalar> q = R.unit_quaternion();

        Serializer<Eigen::Quaternion<Scalar> >::Serialize(stream, q);

        Serializer<Eigen::Matrix<Scalar,3,1> >::Serialize(stream, T.translation());

    }

    static void Deserialize(std::istream & stream,
                            Sophus::SE3<Scalar> & T) {

        Eigen::Quaternion<Scalar> q;
        Eigen::Matrix<Scalar,3,1> t;

        Serializer<Eigen::Quaternion<Scalar> >::Deserialize(stream, q);

        Serializer<Eigen::Matrix<Scalar,3,1> >::Deserialize(stream, t);

        T = Sophus::SE3<Scalar>(q, t);

    }

};

} // namespace internal

template <typename T>
void WriteToStream(std::ostream & stream,
                   const T & value) {

    internal::Serializer<T>::Serialize(stream, value);

}

template <typename Head, typename ... Tails>
void WriteToStream(std::ostream & stream,
                   const Head & headValue,
                   const Tails & ... tailValues) {

    WriteToStream(stream, headValue);

    WriteToStream(stream, tailValues...);

}

template <typename T>
void ReadFromStream(std::istream & stream,
                    T & value) {

    internal::Serializer<T>::Deserialize(stream, value);

}

template <typename Head, typename ... Tails>
void ReadFromStream(std::istream & stream,
                    Head & headValue,
                    Tails & ... tailValues) {

    ReadFromStream(stream, headValue);

    ReadFromStream(stream, tailValues...);

}

template <typename ... Ts>
void WriteToFile(const std::string filename,
                 const Ts & ... values) {

    std::ofstream stream;
    stream.open(filename, std::ios_base::out | std::ios_base::binary);

    WriteToStream(stream, values...);

}

template <typename ... Ts>
void ReadFromFile(const std::string filename,
                  Ts & ... values) {

    if (!pangolin::FileExists(filename)) {
        throw std::runtime_error("file " + filename + " does not exist");
    }

    std::ifstream stream;
    stream.open(filename, std::ios_base::in | std::ios_base::binary);

    ReadFromStream(stream, values...);

}

} // namespace vu
