#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>

namespace py = pybind11;
const double nan_val = std::numeric_limits<double>::quiet_NaN();

py::array_t<double> cal_cpr(py::array_t<int> f_type,
                            py::array_t<double> funds_value)
{
    auto t_buf = f_type.request();
    auto v_buf = funds_value.request();

    const int*    type_ptr  = static_cast<int*>(t_buf.ptr);
    const double* value_ptr = static_cast<double*>(v_buf.ptr);
    const ssize_t D = v_buf.shape[0];
    const ssize_t F = v_buf.shape[1];

    std::unordered_map<int, std::vector<size_t>> type2cols;
    for (size_t c = 0; c < F; ++c)
        type2cols[type_ptr[c]].push_back(c);

    std::vector<std::vector<double>> med(D, std::vector<double>(F, nan_val));
    std::vector<std::thread> med_threads;

    for (const auto& [ftype, cols] : type2cols) {
        med_threads.emplace_back([&, cols]() {
            std::vector<double> buf;
            for (ssize_t d = 0; d < D; ++d) {
                buf.clear();
                for (size_t c : cols) {
                    double v = value_ptr[d * F + c];
                    if (!std::isnan(v)) buf.push_back(v);
                }
                if (buf.empty()) continue;
                std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
                double m = (buf.size() & 1)
                         ? buf[buf.size() / 2]
                         : 0.5 * (*std::max_element(buf.begin(), buf.begin() + buf.size() / 2)
                               + *std::min_element(buf.begin() + buf.size() / 2, buf.end()));
                for (size_t c : cols) med[d][c] = m;
            }
        });
    }
    for (auto& t : med_threads) t.join();

    // -------- 生成比较结果 cmp：线程池并行处理每一行 --------
    std::vector<std::vector<int8_t>> cmp(D, std::vector<int8_t>(F, -1));
    size_t n_cmp_threads = std::min<size_t>(D, std::thread::hardware_concurrency());
    if (n_cmp_threads == 0) n_cmp_threads = 1;
    std::vector<std::thread> cmp_threads(n_cmp_threads);

    for (size_t t = 0; t < n_cmp_threads; ++t) {
        cmp_threads[t] = std::thread([&, t]() {
            for (ssize_t d = t; d < D; d += n_cmp_threads) {
                for (ssize_t c = 0; c < F; ++c) {
                    double v = value_ptr[d * F + c];
                    double m = med[d][c];
                    cmp[d][c] = (std::isnan(v) || std::isnan(m)) ? -1 : (v >= m);
                }
            }
        });
    }
    for (auto& th : cmp_threads) th.join();

    // -------- CPR 计算：列方向并行 --------
    std::vector<double> cpr_list(F, nan_val);
    size_t n_threads = std::min<size_t>(F, std::thread::hardware_concurrency());
    if (n_threads == 0) n_threads = 1;
    std::vector<std::thread> threads(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
        threads[t] = std::thread([&, t]() {
            for (size_t col = t; col < static_cast<size_t>(F); col += n_threads) {
                int ww = 0, ll = 0, wl = 0, lw = 0;
                bool have_last = false;
                int8_t last = 0;
                for (ssize_t d = 0; d < D; ++d) {
                    int8_t cur = cmp[d][col];
                    if (cur == -1) continue;
                    if (!have_last) {
                        last = cur;
                        have_last = true;
                        continue;
                    }
                    if (last == 1 && cur == 1) ww++;
                    else if (last == 0 && cur == 0) ll++;
                    else if (last == 1 && cur == 0) wl++;
                    else if (last == 0 && cur == 1) lw++;
                    last = cur;
                }
                int denom = wl + lw;
                if (denom > 0)
                    cpr_list[col] = double(ww + ll) / denom;
            }
        });
    }
    for (auto& th : threads) th.join();

    return py::array_t<double>(F, cpr_list.data());
}

PYBIND11_MODULE(cal_cpr, m) {
    m.def("cal_cpr", &cal_cpr, "Compute CPR using hand-written thread pool (no SIMD)");
}
