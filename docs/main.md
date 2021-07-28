Documentation
=============
## Introduction ##
Welcome to the Dynamic PET Analysis ToolKit's documentation page!

DPATK bundles convenient routines for the analysis of dynamic PET data using comparmental pharmacokinetic models and decomposition techniques. The toolkit is based on the popular [`numpy`](https://numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-learn`](https://scikit-learn.org/), and [`SimpleITK`](https://simpleitk.org/) libraries.

## License & Citation ##
DPATK is freely available under GPLv3 license, which can be found [here](https://github.com/cormarte/dpatk/blob/main/src/COPYING.txt). If you use DPATK for your publications, please cite the following paper, available [here](https://doi.org/10.3390/cancers13102342):

C. Martens, O. Debeir, C. Decaestecker, T. Metens, L. Lebrun, G. Leurquin-Sterk, N. Trotta, S. Goldman and G. Van Simaeys. Voxelwise Principal Component Analysis of Dynamic [S-Methyl-11C]Methionine PET Data in Glioma Patients. Cancers, 13(10):2342, May 2021.

## Installation ##
### Getting the Sources ###
DPATK's source files can be retrieved from the [github repo](https://github.com/cormarte/dpatk.git) using:

	git clone https://github.com/cormarte/dpatk.git
	
### Install from PyPI ###
Alternately, DPATK can be installed directly from PyPI using:

	pip install dpatk

## Getting Started ##
In this section, the main features of DPATK are presented by means of concrete examples. The entire code of these examples is avaible in the [examples](@ref dpatk.examples) module.

### Reading a Dynamic PET Volume from DICOM Files ###
Most PET scanner vendors will not allow the reconstruction of overlapping frames within a single DICOM series. However, overlapping frames are often a good comprise between temporal resolution and count statistics and can benefit kinetic parameter estimation \cite martens_2021. To deal with this issue, DPATK embeds the convenient class [DICOMDynamicVolumeReader](@ref dpatk.io.DICOMDynamicVolumeReader) for reading and sorting dynamic PET DICOM files possibly belongig to different series but sharing the same Series Date (0008, 0021) and Series Time (0008, 0031) attributes. 

To load a dynamic PET volume from a root directory containing all the DICOM files (possibly in different subdirectories), just type the following instructions:

	from dpatk.io import DICOMDynamicVolumeReader
	
	root_directory = "my/root/directory"
	dynamic_volume, time_array = DICOMDynamicVolumeReader.read(root_directory)
	
A 1D `numpy::ndarray` containing the mid-frame times with regard to the tracer injection time is also returned along with the dynamic PET volume as a [`SimpleITK::Image`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html). 

### Writing a Dynamic PET Volume as a Single MHA File ###
To ease further processing, the sorted dynamic volume can then be stored as a single MHA file using the [SimpleVolumeWriter](@ref dpatk.io.SimpleVolumeWriter):

	from dpatk.io import SimpleVolumeWriter
	
	output_file = "my/output/file.mha"
	SimpleVolumeWriter.write(dynamic_volume, output_file)

### Registering a Dynamic Volume ###
For long dynamic acquisitions, patient motion is likely to occur which can substantially impact the extracted time-activity curves at the voxel level. Therefore, dynamic frame registation is required prior to the analysis. However, in case of short frames, framewise registration may become challenging due to their typically low SNR combined with the rapidly varying contrast in the first minutes after the tracer injection. To address this problem, a block-based approach has been proposed in \cite martens_2021, which consists in gathering frames into blocks for which registration is made easier. Individual frame transforms are then linearly inteprolated from the block transforms. To this end, the [register_dynamic_volume](@ref dpatk.registration.register_dynamic_volume) routine is available in DPATK.

The frame indices of each block must first be specified as a list of ranges:

	block_indices = [range(i, i+30) for i in range(36, 900, 30)]

**Note:** The block indices hereaove were determined for 906 overlapping frames of 20" spaced by 2" as in \cite martens_2021 and must be adjusted according to your specific framing.

The registration can then be launched using:

	registered_dynamic_volume, original_blocks, registered_blocks, parameters = register_dynamic_volume(dynamic_volume, block_indices, delete_original_volume=True)

The 4D registered volume is returned as a [`SimpleITK::Image`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html) along with the original and registered blocks (4D [`SimpleITK::Image`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html)) for quality check and the corresponding registration parameters (2D `numpy::ndarray`).

**Note:** In this example, the `delete_original_volume` flag is set to `True` to save RAM usage. This means that `dynamic_volume` must no longer be used after this call.

### Running a PCA Analysis ###
Principal component analysis (PCA) and other decomposition techniques may be used to extract relevant features from dynamic PET time-activity curves with no need for arterial input function \cite martens_2021. The [models](@ref dpatk.models) module gathers a set of decomposition models for dynamic PET analysis. For example, the [PCAModel](@ref dpatk.models.PCAModel) can be used as follows.

First, a binary 3D mask must be loaded as a [`SimpleITK::Image`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html):

	from dpatk.io import SimpleVolumeReader
	
	mask_file =  "my/mask/file.mha"
	mask = SimpleVolumeReader.read(mask_file)
	
Then, all the time-activity curves must be extracted from the binary mask as a 2D `numpy::ndarray` using the [extract_curves_from_volume](@ref dpatk.utils.extract_curves_from_volume) routine:

	data = extract_curves_from_volume(dynamic_volume, mask, neighborhood=0)/1000.0

**Notes:** 
- A division of the resulting array by 1000 is performed to avoid overflow in the model for data expressed in Bq/ml.
- The `neighborhood` argument is set to 0 in this example, which means that no average smoothing of the curves is performed.

The PCA model is then built using:

	model = Model(nb_of_components=6)
    model.fit(data)
	
and the model parameter (component) values can be computed using:
	
	parameters = model.transform(data)
		
A 4D volume is then rebuilt from the parameter (component) values and the binary mask using the [make_volume_from_curves](@ref dpatk.utils.make_volume_from_curves) routine:

	components_volume = make_volume_from_curves(parameters, mask)
	
The model can finally be saved for further use using the [ModelWriter](@ref dpatk.io.ModelWriter):
	
	from dpatk.io import ModelWriter
	
	model_file = "my/model/file.oby"
	ModelWriter.write(model, model_file)
	
**Note:** For inter-patient comparison of the generated parametric maps, the model should rather be fitted on the whole time-activity curves set extracted from a typical patient dataset after inter-patient normalization (e.g. using SUV conversion) and temporal registration (e.g. using the [register_curve](@ref dpatk.registration.register_curve) routine applied to patient's image-derived input functions). For more information, refer to \cite martens_2021.