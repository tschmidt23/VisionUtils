#pragma once

#include <string>
#include <vector>

namespace vu {

class OptParse {
public:

    // TODO: could probably streamline this w/ a template
    void RegisterOption(const std::string flag, std::string & value, const unsigned char shorthand = -1, const bool required = false);

    void RegisterOption(const std::string flag, int & value, const unsigned char shorthand = -1, const bool required = false);

    void RegisterOption(const std::string flag, float & value, const unsigned char shorthand = -1, const bool required = false);

    void RegisterOption(const std::string flag, double & value, const unsigned char shorthand = -1, const bool required = false);

    void RegisterOption(const std::string flag, bool & value, const unsigned char shorthand = -1, const bool required = false);

    int ParseOptions(int & argc, char * * & argv);

private:

    enum ArgType {
        Integral,
        FloatingPoint,
        DoublePrecisionFloatingPoint,
        Boolean,
        String
    };

    std::vector<std::string> flags_;
    std::vector<unsigned char> shorthands_;
    std::vector<bool> requireds_;
    std::vector<ArgType> types_;
    std::vector<void *> valuePtrs_;

    void RegisterOptionGeneric(const std::string flag, void * valuePtr, const unsigned char shorthand, const bool required, const ArgType type);
};

} // namespace vu
