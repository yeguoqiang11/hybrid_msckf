#ifndef __PERFORMACE_TEST_H__
#define __PERFORMACE_TEST_H__

#include <chrono>

namespace inslam {

class Timer {
public:
    Timer() noexcept : now_(std::chrono::steady_clock::now()) { ; }
    ~Timer() noexcept = default;

    int64_t Pass() const {
        auto pass = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now_).count();
        return pass;
    }

    double Passd() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - now_).count() / 1000.;
    }

    void Reset() { now_ = std::chrono::steady_clock::now(); }

private:
    std::chrono::steady_clock::time_point now_;
};

//#define PERFORMENCE_TEST

#ifdef PERFORMENCE_TEST
    #define TIME_OBJECT_IF_PTEST(c) inslam::Timer c;
    #define TIME_PASS_IF_PTEST(c, t) auto t = c.Passd();
#else 
    #define TIME_OBJECT_IF_PTEST(c) 
    #define TIME_PASS_IF_PTEST(c, t)
#endif

}//namespace inslam {

#endif//__PERFORMACE_TEST_H__