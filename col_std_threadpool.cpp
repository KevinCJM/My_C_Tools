#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <thread>

namespace py = pybind11;

void compute_column_std(double* ptr, size_t rows, size_t cols, size_t j, double* result) {
    double sum = 0.0, sum_sq = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < rows; ++i) {
        double val = ptr[i * cols + j];
        if (!std::isnan(val)) {
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    if (count > 1) {
        double mean = sum / count;
        result[j] = std::sqrt((sum_sq - count * mean * mean) / (count - 1));
    } else {
        result[j] = std::nan("");
    }
}

py::array_t<double> column_std_threadpool(py::array_t<double> input) {
    auto buf = input.request();
    size_t rows = buf.shape[0], cols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> result(cols, std::nan(""));
    size_t n_threads = std::min(cols, (size_t)std::thread::hardware_concurrency());
    std::vector<std::thread> threads(n_threads);
    for (size_t t = 0; t < n_threads; ++t) {
        threads[t] = std::thread([=, &result]() {
            for (size_t j = t; j < cols; j += n_threads)
                compute_column_std(ptr, rows, cols, j, result.data());
        });
    }
    for (auto& t : threads) t.join();
    return py::array_t<double>(result.size(), result.data());
}

PYBIND11_MODULE(core, m) {
    m.def("column_std", &column_std_threadpool, "Compute column-wise std (ddof=1, ignore NaNs)");
}
