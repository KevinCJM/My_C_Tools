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
        f'my_ctools.{mod}',
        [f"my_ctools/{mod}.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    )
    for mod in cpp_modules
]

setup(
    name='my-ctools',
    version='0.1.5',
    author='KevinCJM',
    description='Fast column-wise std calculator using C++ thread pool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['my_ctools'],
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.6',
)
