#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timestamp;

#define GET_3RD_ARG(arg1, arg2, arg3, ...) arg3

#define TIMER_INIT2(timer_name, template_type) std::chrono::duration<template_type> timer_name
#define TIMER_INIT1(timer_name) TIMER_INIT2(timer_name, double)
#define TIMER_INIT_CHOOSER(...) GET_3RD_ARG(__VA_ARGS__, TIMER_INIT2, TIMER_INIT1, )
#define TIMER_INIT(...) TIMER_INIT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#define TIME_IT2(timer_name, template_type) \
    for (struct { int dummy = 0; timestamp start = Clock::now(); } timer_name##_struct; \
         timer_name##_struct.dummy < 1; \
         timer_name = std::chrono::duration_cast<template_type>(Clock::now() - timer_name##_struct.start),\
         timer_name##_struct.dummy++)
#define TIME_IT1(timer_name) TIME_IT2(timer_name, std::chrono::duration<double>)
#define TIME_IT_CHOOSER(...) GET_3RD_ARG(__VA_ARGS__, TIME_IT2, TIME_IT1, )
#define TIME_IT(...) TIME_IT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

struct Timer {
    timestamp prev;
    template <typename T = std::chrono::duration<double>>
    T tick() {
        const auto end = Clock::now();
        const T diff = std::chrono::duration_cast<T>(end - this->prev);
        this->prev = end;
        return diff;
    }
};

struct Measure {
    size_t count;
    std::chrono::duration<double> total, min, max;
    Measure()
    : count(0),
    total(std::chrono::duration<double>::zero()),
    min(std::chrono::duration<double>::max()),
    max(std::chrono::duration<double>::min())
    {}
    void addSample(std::chrono::duration<double> time) {
        count++;
        total += time;
        if (min > time) min = time;
        if (max < time) max = time;
    }
};

inline std::ostream& operator<<(std::ostream& stream, std::chrono::duration<double> d) {
    if (d.count() < 0.000001)
        return stream << d.count()*1000000000 << "ns";
    if (d.count() < 0.001)
        return stream << d.count()*1000000 << "us";
    if (d.count() < 1)
        return stream << d.count()*1000 << "ms";
    return stream << d.count() << "s";
}
inline std::ostream& operator<<(std::ostream& stream, std::chrono::milliseconds d) {
    return stream << d.count() << "ms";
}
inline std::ostream& operator<<(std::ostream& stream, std::chrono::microseconds d) {
    return stream << d.count() << "us";
}
inline std::ostream& operator<<(std::ostream& stream, Measure m) {
    return stream << m.total << " \t" << (m.total/m.count) << " \t" << m.min << " \t" << m.max;
}

#endif
