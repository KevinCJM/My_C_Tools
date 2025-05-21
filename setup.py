from setuptools import setup, Extension
import pybind11

cpp_modules = [
    "cal_std_mean",
    "cal_cpr",
    "cal_all_largest_indicators",
    "cal_all_longest_indicators",
    "cal_longest_dd_recover",
    "cal_max_dd",
    "cal_rolling_gain_loss",
]

ext_modules = [
    Extension(
        f"my_ctools.{mod}",
        [f"my_ctools/{mod}.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fPIC", "-mavx"],
    )
    for mod in cpp_modules
]

setup(
    name="my_ctools",
    version="0.1.8",
    author="KevinCJM",
    description="Fast C++ indicators for fund analytics",
    packages=["my_ctools"],
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.8",
)
