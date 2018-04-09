#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <iostream>
#include <map>
#include <string>

#include <pangolin/utils/timer.h>

namespace vu {

// GlobalTimer is a Myers singleton as described in Modern C++ Design 6.4
// It therefore has the Dead Reference problem (described in  6.5), but
// there are no dependencies so this should be OK.
class GlobalTimer {
public:

    static GlobalTimer & GetTimer() {
        static GlobalTimer timer;
        return timer;
    }

    static inline void Tick(const std::string name) {
        GetTimer().Tick_(name);
    }

    static inline void Tock(const std::string name) {
        GetTimer().Tock_(name);
    }

private:

    GlobalTimer() { }
    GlobalTimer(const GlobalTimer &);
    GlobalTimer & operator=(const GlobalTimer &);

    inline void Tick_(const std::string name) {

        if (timings_.find(name) == timings_.cend()) {

            timings_[name] = { 0.0, 0, pangolin::TimeNow() };

        } else {

            timings_[name].start_ = pangolin::TimeNow();

        }

    }

    inline void Tock_(const std::string name) {

        pangolin::basetime end = pangolin::TimeNow();

        ClockInfo & clock = timings_[name];

        const double timeDiff = pangolin::TimeDiff_us(clock.start_,end);

        ++clock.count_;

        clock.totalTime_ += timeDiff;

    }

    ~GlobalTimer() {

        for (std::pair<const std::string,ClockInfo> & timing : timings_) {

            const double averageTime = timing.second.totalTime_ / timing.second.count_;

            std::cout << timing.first << ": " << (averageTime*0.001) << " ms on average (" << timing.second.count_ << " calls)" << std::endl;

        }

    }

    struct ClockInfo {
        double totalTime_;
        int count_;
        pangolin::basetime start_;
    };

    std::map<std::string,ClockInfo> timings_;

};

} // namespace vu
