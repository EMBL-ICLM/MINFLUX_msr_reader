from setuptools import setup, find_packages
import os

# Get all DLL and PYD files from the specpy subdirectory
specpy_dir = os.path.join(os.path.dirname(__file__), 'specpy')
data_files = []
if os.path.exists(specpy_dir):
    for fname in os.listdir(specpy_dir):
        if fname.endswith(('.pyd', '.dll')):
            data_files.append(fname)

setup(
    name="specpy",
    version="1.2.3",
    package_dir={'': 'specpy'},
    py_modules=["specpy", "version"],
    package_data={
        '': data_files,
    },
    include_package_data=True,
    description="SpecPy library for MINFLUX data analysis",
)
