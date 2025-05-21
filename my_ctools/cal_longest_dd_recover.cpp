// cal_longest_dd_recover.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <limits>
#include <cmath>

namespace py = pybind11;
const double NaN = std::numeric_limits<double>::quiet_NaN();

/* ------------------------------------------------------------------ */
/* 计算单列最长 drawdown-recover 段长度（与 Python 完全一致）          */
static long longest_segment_len(const double* data,
                                ssize_t rows, ssize_t cols, ssize_t col)
{
    /* 1) np.nancumprod + prepend 1 ----------------------------------- */
    std::vector<double> net(rows + 1);
    net[0] = 1.0;
    double acc = 1.0;
    for (ssize_t r = 0; r < rows; ++r) {
        double v = data[r * cols + col];
        if (!std::isnan(v))         // NaN → 0% 视作 1
            acc *= (1.0 + v);
        net[r + 1] = acc;
    }

    /* 2) running_max & drawdown != 0 -------------------------------- */
    double run_max = net[0];
    long cur = 0, longest = 0;
    for (ssize_t i = 1; i <= rows; ++i) {
        run_max = std::max(run_max, net[i]);
        double dd = (net[i] - run_max) / run_max;   // ≤ 0
        bool in_dd = (dd != 0.0);                   // 精确对齐 Python (!= 0)
        if (in_dd) {
            ++cur;
            if (cur >= longest) longest = cur;      // 取“最后一个”最长段
        } else {
            cur = 0;
        }
    }
    return longest;
}

/* ================================================================== */
py::array_t<long> cal_longest_draw_down_recover(py::array_t<double> funds_val)
{
    auto buf = funds_val.request();
    const ssize_t R = buf.shape[0];     // 天
    const ssize_t C = buf.shape[1];     // 基金
    const double* p = static_cast<double*>(buf.ptr);

    std::vector<long> result(C, 0);

    /* 多线程按列并行 ----------------------------------------------- */
    std::size_t n_th = std::min<std::size_t>(C,
                          std::thread::hardware_concurrency());
    if (n_th == 0) n_th = 1;
    std::vector<std::thread> ths(n_th);

    for (std::size_t tid = 0; tid < n_th; ++tid) {
        ths[tid] = std::thread([&, tid]() {
            for (ssize_t c = tid; c < C; c += n_th)
                result[c] = longest_segment_len(p, R, C, c);
        });
    }
    for (auto& t : ths) t.join();

    return py::array_t<long>(C, result.data());
}

/* ------------------------------------------------------------------ */
PYBIND11_MODULE(cal_longest_dd_recover, m)
{
    m.def("cal_longest_dd_recover",
          &cal_longest_draw_down_recover,
          "Longest drawdown-recovery days per fund (C++ & thread-parallel)");
}
