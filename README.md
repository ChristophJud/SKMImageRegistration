# SKM - Sparse Kernel Machine for Image Registration
SKM is an image registration software written in `C++11` and implements the kernel machine for image registration published in `Jud et al. [2016a, 2016b]`. The computational heavy transformation model is accelerated by executing it on the graphics processing unit where the parallel computing platform CUDA is used.

## Getting Started
The software provides an image registration functionality in the spirit of the [ITK](https://itk.org/) registration framework (v3), where the computational heavy transformation, regularization, value and derivative computations are performed on the GPU. It can be included as a library into your own source code or be executed on the command line through the provided `SKMReg` application. The software can be basically built for either 2d or 3d images which can be specified in the build process (see Installation).

### Prerequisities
The project has been tested with the following hardware/software configurations on
Ubuntu 16.04

#### Hardware
* NVIDIA GeForce GTX 970
* NVIDIA GeForce GTX 1080
* NVIDIA GeForce GTX TITAN X

#### Build Tools
* gcc 4.9.3
* cmake/ccmake 3.5.1
* git 2.7.4

#### Packages and Drivers
* Nvidia Cuda 7.5 or 8.0 _(Ubuntu packages: nvidia-cuda-dev nvidia-cuda-toolkit)_
* Nvidia device drivers _(Ubuntu package: nvidia-361)_
* HDF5 _(Ubuntu packages: libhdf5-cpp-11 libhdf5-dev)_

### Installation
```sh
git clone https://github.com/ChristophJud/SKMImageRegistration.git
cd SKMImageRegistration
mkdir build
mkdir install
cd build
ccmake ../superbuild
```
* Configure by pressing 'c'
* Set the number of space dimensions of your images (2 or 3)
* Set the installation directory `CMAKE_INSTALL_PREFIX` (e.g. `<project_dir>/install`)
* Set the installation directory of the dependencies `INSTALL_DEPENDENCIES_DIR` (e.g. `<project_dir>/install/thirdparty`)
* Reconfigure and generate by pressing 'g'

Compile the project:
```sh
make -j<num_cores - 1>
```
The project is automatically installed in the provided directories, thus no rule to make an install target is defined. The `SKMReg` application can be found in `<install_directory>/bin`.
#### Dependencies
The project depends on the [Insight Toolkit (ITK)](https://itk.org/), [Json for Modern C++](https://github.com/nlohmann/json) and the [BPrinter](https://github.com/dattanchu/bprinter) library, which are automatically built as external projects when running the `cmake` superbuild.

### Running an Example
Basically, there are three different sources of parameter values SKM uses. The first one are the default values which are hardcoded in the software (see `ConfigUtils.h`). As a second source, parameter values can be provided through a `json` file and finally each parameter can be also set by a command line argument. The different sources overwrite the values in the mentioned order.

Usual calls of SKM looks as follows:
```
SKMReg help
SKMReg version
SKMReg config_file /tmp/config.json
SKMReg config_file /tmp/config.json temp_directory /tmp/experiment
SKMReg reference_filename /tmp/ref.mhd target_filename /tmp/tar.mhd metric \"ncc\" reg_l1 "[1e-2, 1e-3, 1e-4]"
```
SKM writes severel intermediate results and the final results to the `temp_directory` which should be created before calling SKM. The default location is `/tmp/skm`. The final results are:
* Displacement field: `<temp_directory>/df.vtk`
* Warped reference image: `<temp_directory>/warped.vtk`

To look at the results and visualize them, you can e.g. use [ParaView](http://www.paraview.org/).
## History
The project started in the [Medical Image Analysis Center](http://dbe.unibas.ch:8080/magnoliaPublic/dbe/research/ResearchGroups/miac.html) research group of the [University of Basel](http://www.unibas.ch), where we had to efficiently register abdominal image time series. 
##### Authors and Contributors
* **Christoph Jud** - *initial work* _(christoph.jud@unibas.ch)_
* **Benedikt Bitterli** - *cuda support*
* **Nadia Möri** - *mathematical support*
* **Robin Sandkühler** - *coding support*
* **Philippe C. Cattin** - *project support*

## License
SKM is licensed under the Apache 2.0 license. For details, consider the LICENSE and NOTICE file.

If you can use this software in any way, please cite us in your publications:

[2016a] Christoph Jud, Nadia Möri, and Philippe C. Cattin. "Sparse Kernel Machines for Discontinuous Registration and Nonstationary Regularization". In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pages 9-16, 2016. [link](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w15/papers/Jud_Sparse_Kernel_Machines_CVPR_2016_paper.pdf)

and/or

[2016b] Christoph Jud, Nadia Möri, Benedikt Bitterli and Philippe C. Cattin. "Bilateral Regularization in Reproducing Kernel Hilbert Spaces for Discontinuity Preserving Image Registration". In *7th International Conference on Machine Learning in Medical Imaging*, 2016.

### Contributing
We released SKM to contribute to the community. Thus, if you find and/or fix bugs or extend the software please contribute as well and let us know or make a pull request. 

### Other Open Source Projects
SKM depends on several thirdparty open source project which are either linked as library or has been directly integrated into the source code. For details, consider the NOTICE file.
