#include "noa/cpu/fourier/Plan.h"

using namespace Noa::Fourier::Details;

std::mutex Mutex::mutex;

fftwf_plan PlansBuffer::m_plans_float[MAX_PLAN]{};
fftw_plan PlansBuffer::m_plans_double[MAX_PLAN]{};
uint PlansBuffer::m_index_float{0};
uint PlansBuffer::m_index_double{0};

void PlansBuffer::push(fftwf_plan plan) {
    if (m_index_float < MAX_PLAN) {
        fftwf_plan old_plan = m_plans_float[m_index_float];
        if (old_plan) {
            std::unique_lock<std::mutex> lock(Details::Mutex::get());
            fftwf_destroy_plan(old_plan);
        }
        m_plans_float[m_index_float] = plan;
    } else {
        m_index_float = 0;
        push(plan);
        return;
    }
    m_index_float++;
}

void PlansBuffer::push(fftw_plan plan) {
    if (m_index_double < MAX_PLAN) {
        if (m_plans_double[m_index_double]) {
            std::unique_lock<std::mutex> lock(Details::Mutex::get());
            fftw_destroy_plan(m_plans_double[m_index_double]);
        }
        m_plans_double[m_index_double] = plan;
    } else {
        m_index_double = 0;
        push(plan);
        return;
    }
    m_index_double++;
}

void PlansBuffer::clear_plans_float() {
    std::unique_lock<std::mutex> lock(Details::Mutex::get());
    for (uint idx{0}; idx < MAX_PLAN; ++idx) {
        if (m_plans_float[idx]) {
            fftwf_destroy_plan(m_plans_float[idx]);
            m_plans_float[idx] = nullptr;
        }
    }
}

void PlansBuffer::clear_plans_double() {
    std::unique_lock<std::mutex> lock(Details::Mutex::get());
    for (uint idx{0}; idx < MAX_PLAN; ++idx) {
        if (m_plans_double[idx]) {
            fftw_destroy_plan(m_plans_double[idx]);
            m_plans_double[idx] = nullptr;
        }
    }
}
