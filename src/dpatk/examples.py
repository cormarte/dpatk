# This file is part of the dpatk package.
# Copyright (C) 2021  Corentin Martens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Contact: corentin.martens@ulb.be

import argparse
import sys
from dpatk.io import DICOMDynamicVolumeReader, SimpleVolumeReader, SimpleVolumeWriter
from dpatk.models import FAModel, PCAModel
from dpatk.registration import register_dynamic_volume
from dpatk.utils import extract_curves_from_volume, make_volume_from_curves
from os.path import dirname, join


def read_write_volume(input_directory, output_file):

    """
    Reads a 4D dynamic PET volume sorted as DICOM files possibly belonging to different series with overlapping
    frames and stores it as a single MHA file.

    Parameters
    ----------
    input_directory : str
        The root directory containing all dynamic PET DICOM files for the series. Will be walked recursively.
    output_file : str
        The path of the output dynamic PET file.

    """

    # First, let us load the dynamic volume from the directory as a SimpletITK::Image.
    # An array containing the mid-frame times wrt the tracer injection is also returned.
    dynamic_volume, time_array = DICOMDynamicVolumeReader.read(input_directory)

    # Then, the sorted dynamic volume is stored as a single MHA file to ease further processing.
    SimpleVolumeWriter.write(dynamic_volume, output_file)


def register_volume(input_file, output_file):

    """
    Rigidly registers all frames of a 4D dynamic PET volume by mutual information maximization.

    More information on the method used can be found in Martens et al. Cancers 13(10):2342. 2021.

    Parameters
    ----------
    input_file : str
        The path of the dynamic PET file in MHA format.
    output_file : str
        The path of the output registered dynamic PET file.

    """

    # First, let us load the sorted dynamic volume stored as a single file.
    dynamic_volume = SimpleVolumeReader.read(input_file)

    # Then, let us define the frame indices of each block for the registration as a list of ranges.
    # Default values for 906 overlapping frames of 20" spaced by 2". Must be adjusted for each specific framing.
    block_indices = [range(i, i+30) for i in range(36, 900, 30)]

    # Let us now launch the registration. The 4D registered volume is returned as a SimpletITK::Image along with the
    # original and registered blocks (4D SimpletITK::Image) for quality check and the corresponding registration
    # parameters (2D numpy::ndarray).
    # In this example, the delete_original_volume flag is set to true to save RAM usage. This means that dynamic_volume
    # must no longer be used after this call.
    registered_dynamic_volume, original_blocks, registered_blocks, parameters = \
        register_dynamic_volume(dynamic_volume, block_indices, delete_original_volume=True)

    # Finally, let us save the results as single MHA files.
    SimpleVolumeWriter.write(registered_dynamic_volume, output_file)
    SimpleVolumeWriter.write(original_blocks, join(dirname(output_file), 'original_blocks.mha'))
    SimpleVolumeWriter.write(registered_blocks, join(dirname(output_file), 'registered_blocks.mha'))


def run_decomposition_analyses(dynamic_volume_file, mask_file):

    """
    Runs the decomposition analyses (FA and PCA) of a 4D dynamic PET volume within a 3D binary mask.

    Parameters
    ----------
    dynamic_volume_file : str
        The path of the dynamic PET file in MHA format.
    mask_file : str
        The path of the binary mask file in MHA format.

    """

    # First, let us load the registered dynamic volume and the binary mask stored as single files.
    dynamic_volume = SimpleVolumeReader.read(dynamic_volume_file)
    mask = SimpleVolumeReader.read(mask_file)

    # Then, extract all curves within the mask from the dynamic volume as a 2D numpy::ndarray.
    # A division of the resulting array by 1000 is performed to avoid overflow for data expressed in Bq/ml".
    # The neighborhood argument is set to 0, which means that no average smoothing of the curves is performed.
    data = extract_curves_from_volume(dynamic_volume, mask, neighborhood=0)/1000.0

    # Define the desired number of components for the decomposition models.
    nb_of_components = 6

    # Define the desired decomposition models.
    models = [FAModel, PCAModel]

    for Model in models:

        # Instantiate the model and fit the data.
        model = Model(nb_of_components=nb_of_components)
        model.fit(data)

        # Transform the curves into model parameter (component) values.
        parameters = model.transform(data)

        # Rebuild a 4D volume from the parameter (component) values and the binary mask.
        components_volume = make_volume_from_curves(parameters, mask)

        # Finally, let us write the resulting volume as a single MHA file.
        SimpleVolumeWriter.write(components_volume, join(dirname(dynamic_volume_file),
                                                         f'{model.get_name().lower()}.mha'))


def run_all_examples(directory):

    """
    Successively runs all the examples from a root DICOM directory.

    Notes: - A 3D binary mask with the same dimensions as the dynamic PET volume must be drawn and stored in MHA format
             in the root directory prior to the analysis.
           - The default frame indices of the blocks used for dynamic volume registration were determined for 906
             overlapping frames of 20" spaced by 2" and must be adjusted according to the specific framing.

    Parameters
    ----------
    directory : str
        The root directory containing the dynamic PET DICOM files and the MHA binary mask file.

    """

    read_write_volume(directory, join(directory, 'volume.mha'))
    register_volume(join(directory, 'volume.mha'), join(directory, 'registered_volume.mha'))
    run_decomposition_analyses(join(directory, 'registered_volume.mha'), join(directory, 'mask.mha'))


def main(argv):

    parser = argparse.ArgumentParser(description='DPATK Examples')
    parser.add_argument('--directory', dest='directory', type=str,
                        help='The root directory containing the dynamic PET DICOM files and the MHA binary mask file.',
                        required=True)
    args = parser.parse_args(argv)
    run_all_examples(args.directory)


if __name__ == '__main__':

    main(sys.argv[1:])
