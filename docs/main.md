Documentation
=============
## Introduction ##
Welcome to the Dynamic PET Analysis ToolKit's documentation page!

DPATK bundles convenient routines for the analysis of dynamic PET data using comparmental pharmacokinetic models and decomposition techniques. The toolkit is based on the popular [`numpy`](https://numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-learn`](https://scikit-learn.org/), and [`SimpleITK`](https://simpleitk.org/) packages.

## License & Citation ##
DPATK is freely available under GPLv3 license, which can be found [here](https://github.com/cormarte/dpatk/blob/main/src/COPYING.txt). If you use DPATK for your publications, please cite the following paper, available [here](https://doi.org/10.3390/cancers13102342):

C. Martens, O. Debeir, C. Decaestecker, T. Metens, L. Lebrun, G. Leurquin-Sterk, N. Trotta, S. Goldman and G. Van Simaeys. Voxelwise Principal Component Analysis of Dynamic [S-Methyl-11C]Methionine PET Data in Glioma Patients. Cancers, 13(10):2342, May 2021.

## Installation ##
### Getting the Sources ###
DPATK's source files can be retrieved from the [github repo](https://github.com/cormarte/dpatk.git) using:

	git clone https://github.com/cormarte/dpatk.git
	
### Install DPATK from PyPI ###
Alternately, DPATK can be installed directly from PyPI using:

	pip install dpatk

## Getting Started ##
### Reading a Dynamic PET Volume from DICOM files ###
Most PET scanner vendors will not allow the reconstruction of overlapping frames within a single DICOM series. However, overlapping frames lead to a good comprise between temporal resolution and count statistics and can benefit kinetic parameter estimation \cite martens_2021. To deal with this issue, DPATK bundles the convenient class [DICOMDynamicVolumeReader](@ref dpatk.io.DICOMDynamicVolumeReader) for reading and sorting dynamic PET DICOM files possibly belongig to different series but sharing the same Series Date (0008, 0021) and Series Time (0008, 0031) attributes. To load a dynamic PET volume from a root directory containing all the DICOM files, just type:

	from dpatk.io import DICOMDynamicVolumeReader
	
	root_directory = "my/root/directory"
	dynamic_volume, time_array = DICOMDynamicVolumeReader.read(root_directory)
	
A 1D `numpy::ndarray` containing the mid-frame times with regard to the tracer injection is also returned along with the dynamic PET volume as a `SimpleITK::Image`. To ease further processing, the sorted dynamic volume can then be stored as a single MHA file using the [SimpleVolumeWriter](@ref dpatk.io.SimpleVolumeWriter):

	from dpatk.io import SimpleVolumeWriter
	
	output_file = "my/output/file.mha"
	SimpleVolumeWriter.write(dynamic_volume, output_file)
