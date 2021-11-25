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

import numpy as np
import pickle
import pydicom
import SimpleITK as sitk
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from dpatk.models import ModelBase
from numpy import ndarray
from os import walk
from os.path import isdir, isfile, join
from SimpleITK import Image
from sortedcontainers import SortedDict


class ReaderBase(ABC):

    """
    Base class for data readers.
    """

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def read(path: str):

        """Reads the data.

        Parameters
        ----------
        path : str
            A file path.

        Returns
        ----------
        data : object
            The loaded data.

        """
        pass


class DICOMDynamicVolumeReader(ReaderBase):

    """
    A reader for 4D dynamic PET volumes sorted as DICOM files possibly belonging to different series with overlapping
    frames.
    """

    def __init__(self):

        super().__init__()

    @staticmethod
    def _get_dynamic_pet_volume(sorted_files: SortedDict):

        """Reads a 4D dynamic PET volume as an Image from a SortedDict with each frame's Frame Reference Time
        (0054, 1300) attribute as keys and a list of corresponding DICOM files as values, sorted with regard to their
        respective Image Position (Patient) (0020, 0032) z-coordinate.

        Parameters
        ----------
        sorted_files : SortedDict
            A SortedDict with, for each frame, its Frame Reference Time (0054, 1300) attribute as key and as value, a
            list of the corresponding files sorted with regard to their respective Image Position (Patient) (0020, 0032)
            z-coordinate.

        Returns
        -------
        dynamic_volume : Image
            The 4D dynamic PET volume in output units.
        time_array : ndarray
            A 1D array specifying the mid-frame time in seconds for each frame of dynamic_volume wrt the injection time.

        """

        series_reader = sitk.ImageSeriesReader()
        series_reader.SetGlobalWarningDisplay(False)
        series_reader.SetLoadPrivateTags(True)
        series_reader.SetMetaDataDictionaryArrayUpdate(True)

        reference_files = list(sorted_files.values())[0]
        series_reader.SetFileNames(reference_files)
        reference_image = series_reader.Execute()

        series_date = series_reader.GetMetaData(0, '0008|0021')
        series_time = series_reader.GetMetaData(0, '0008|0031')
        series_datetime = series_date + series_time

        # SimpleITK cannot read DICOM SQ elements
        # TODO: Remove extra dependency to pydicom
        meta_data = pydicom.dcmread(reference_files[0], stop_before_pixels=True)

        if [0x0054, 0x0016] in meta_data:
            radiopharmaceutical_sequence = meta_data[0x0054, 0x0016][0]
            if [0x0018, 0x1078] in radiopharmaceutical_sequence:
                injection_datetime = radiopharmaceutical_sequence[0x0018, 0x1078].value
            elif [0x0018, 0x1072] in radiopharmaceutical_sequence:
                injection_datetime = series_date + radiopharmaceutical_sequence[0x0018, 0x1072].value
            else:
                injection_datetime = series_datetime
                warnings.warn('Injection time could not be retrieved. Series time will be used.')
        else:
            injection_datetime = series_datetime
            warnings.warn('Injection time could not be retrieved. Series time will be used.')

        if series_reader.HasMetaDataKey(0, '0008|0070'):
            manufacturer = series_reader.GetMetaData(0, '0008|0070').lower()
        else:
            manufacturer = 'unknown'
            warnings.warn('Unknown manufacturer. Frame Reference Time will be considered relative to mid-frame.')

        series_time = datetime.strptime(series_datetime, '%Y%m%d%H%M%S.%f')
        injection_time = datetime.strptime(injection_datetime, '%Y%m%d%H%M%S.%f')
        injection_delay = (injection_time - series_time).total_seconds() * 1000.0

        dimensions = reference_image.GetSize()
        number_of_frames = len(sorted_files)

        dynamic_pet_array = np.zeros((dimensions[2], dimensions[1], dimensions[0], number_of_frames), dtype=np.float64)
        time_array = np.zeros(number_of_frames, dtype=np.float64)
        index = 0

        for reference_time, files in sorted_files.items():

            series_reader.SetFileNames(files)
            image = series_reader.Execute()

            mid_frame_time = reference_time - injection_delay
            # GE's Frame Reference Time (0054, 1300) refers to the start of the frame, so the half Actual Frame Duration
            # (0018, 1242) is added
            if 'ge' in manufacturer and series_reader.HasMetaDataKey(0, '0018|1242'):
                mid_frame_time += 0.5*(float(series_reader.GetMetaData(0, '0018|1242')))

            time_array[index] = mid_frame_time/1000.0
            dynamic_pet_array[:, :, :, index] = sitk.GetArrayViewFromImage(image)
            index += 1

        dynamic_volume = sitk.GetImageFromArray(dynamic_pet_array)
        dynamic_volume.CopyInformation(reference_image)

        return dynamic_volume, time_array

    @staticmethod
    def _get_sorted_dynamic_pet_dicom_files(directory: str):

        """Sorts dynamic PET DICOM files from a root directory.

        Overlapping frames for a given Study Instance UID (0020, 000D) reconstructed in different series will be
        gathered provided that they share the same Series Date (0008, 0021) and Series Time (0008, 0031) attributes.

        Note that only images of Modality (0008, 0060) 'PT' and Series Type (0054, 1000) 'DYNAMIC\\IMAGE' will be
        considered.

        Parameters
        ----------
        directory : str
            The root directory containing all dynamic PET DICOM files for the series. Will be walked recursively.

        Returns
        -------
        sorted_files : list of SortedDict
            A list containing, for each Study Instance UID/Series Datetime combination, a SortedDict with each found
            frame's Frame Reference Time (0054, 1300) attribute as key and as value a list of the corresponding files
            sorted wrt their respective Image Position (Patient) (0020, 0032) z-coordinate.

        """

        file_reader = sitk.ImageFileReader()
        sorted_dict = {}

        for root, dirs, files in walk(directory, topdown=False):
            for file in files:
                path = join(root, file)
                file_reader.SetFileName(path)
                file_reader.ReadImageInformation()
                try:
                    modality = file_reader.GetMetaData('0008|0060')
                    if 'PT' in modality:
                        series_type = file_reader.GetMetaData('0054|1000')
                        if 'DYNAMIC\\IMAGE' in series_type:
                            study_id = file_reader.GetMetaData('0020|000d')
                            series_datetime = file_reader.GetMetaData('0008|0021') + \
                                              file_reader.GetMetaData('0008|0031')
                            reference_time = float(file_reader.GetMetaData('0054|1300'))
                            slice_location = float(file_reader.GetMetaData('0020|0032').split('\\')[-1])
                            if study_id not in sorted_dict:
                                sorted_dict[study_id] = {}
                            if series_datetime not in sorted_dict[study_id]:
                                sorted_dict[study_id][series_datetime] = SortedDict()
                            if reference_time not in sorted_dict[study_id][series_datetime]:
                                sorted_dict[study_id][series_datetime][reference_time] = SortedDict()
                            sorted_dict[study_id][series_datetime][reference_time][slice_location] = path
                except RuntimeError:
                    warnings.warn(f'Required metadata could not be read for file \'{path}\'. File will be ignored.')

        sorted_files = []

        for study_id in sorted_dict.keys():
            for series_datetime in sorted_dict[study_id].keys():
                series_dict = SortedDict()
                for reference_time in sorted_dict[study_id][series_datetime].keys():
                    series_dict[reference_time] = list(sorted_dict[study_id][series_datetime][reference_time].values())
                sorted_files.append(series_dict)

        return sorted_files

    @staticmethod
    def read(path: str):

        """Reads dynamic PET DICOM files from a root directory into a single 4D dynamic PET volume.

        Overlapping frames reconstructed in different series will be merged provided that they share the same Series
        Date (0008, 0021) and Series Time (0008, 0031) attributes.

        If more than one dynamic volume is found in the root directory, the first one will be returned.

        Parameters
        ----------
        path : str
            The root directory containing all dynamic PET DICOM files for the series. Will be walked recursively.

        Returns
        -------
        dynamic_volume : Image
            The 4D merged dynamic PET volume in output units.
        time_array : ndarray
            A 1D array specifying the mid-frame time in seconds for each frame of dynamic_volume w.r.t. the injection
            time.

        """

        assert isdir(path)

        sorted_files = DICOMDynamicVolumeReader._get_sorted_dynamic_pet_dicom_files(path)

        if len(sorted_files) == 0:
            raise RuntimeError('No dynamic PET volume found in the specified directory.')

        elif len(sorted_files) > 1:
            warnings.warn('More than one dynamic PET volume found in the specified directory, returning the first one.')

        dynamic_volume, time_array = DICOMDynamicVolumeReader._get_dynamic_pet_volume(sorted_files[0])

        return dynamic_volume, time_array


class ModelReader(ReaderBase):

    """
    A reader for ModelBase objects.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def read(path: str):

        """Reads a model.

        Parameters
        ----------
        path : str
            A file path.

        Returns
        ----------
        model : ModelBase
            The loaded model.

        """

        assert isfile(path)

        model = pickle.load(open(path, 'rb'))

        return model


class SimpleVolumeReader(ReaderBase):

    """
    A reader for medical image volumes stored as single files.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def read(path: str):

        """Reads an image volume.

        The file format will be determined by SimpleITK.ReadImage based on the file extension if supported.

        Parameters
        ----------
        path : str
            A file path.

        Returns
        ----------
        volume : Image
            The loaded image volume.

        """

        assert isfile(path)

        volume = sitk.ReadImage(path)

        return volume


class WriterBase(ABC):

    """
    Base class for data writers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def write(self, data: object, path: str):

        """Writes the data.

        Parameters
        ----------
        data : object
            The data.
        path : str
            The file path.

        """
        pass


class DICOMVolumeWriter(WriterBase):

    """
    A writer for medical image volumes stored as DICOM.
    """

    def __init__(self, ref_file):
        super().__init__()
        self.ref_file = ref_file

    def write(self, data: Image, path: str):

        """Writes an image volume.

        Parameters
        ----------
        data : Image
            An image volume.
        path : str
            The DICOM directory.

        """

        # Reference file reading
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.ref_file)
        reader.ReadImageInformation()

        # Image information
        direction = data.GetDirection()
        spacing = data.GetSpacing()
        number_of_slices = data.GetDepth()
        number_of_components = data.GetNumberOfComponentsPerPixel()

        # Image rescaling
        array = sitk.GetArrayViewFromImage(data)
        input_min, input_max = np.min(array), np.max(array)
        output_type = np.int16
        output_min, output_max = np.iinfo(output_type).min, np.iinfo(output_type).max
        slope = (input_max-input_min)/(output_max-output_min)
        intercept = -output_min*slope+input_min

        # Tags
        tags_to_copy = ['0008|0020',  # Study Date
                        '0008|0030',  # Study Time
                        '0008|0050',  # Accession Number
                        '0008|0060',  # Modality
                        '0010|0010',  # Patient Name
                        '0010|0020',  # Patient ID
                        '0010|0030',  # Patient Birth Date
                        '0020|000d',  # Study Instance UID
                        '0020|0010'   # Study ID
                        ]

        modification_time = time.strftime('%H%M%S')
        modification_date = time.strftime('%Y%m%d')

        series_tag_values = [(t, reader.GetMetaData(t)) for t in tags_to_copy if reader.HasMetaDataKey(t)] + \
                            [('0008|0008', 'DERIVED\\SECONDARY'),                                                       # Image Type
                             ('0008|0021', modification_date),                                                          # Series Date
                             ('0008|0031', modification_time),                                                          # Series Time
                             ('0008|103e', f'{reader.GetMetaData("0008|103e")} - DPATK Processed'),                     # Series Description
                             ('0020|000e', f'1.2.826.0.1.3680043.2.1125.{modification_date}.1{modification_time}'),     # Series Instance UID
                             ('0020|0037', '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1],
                                                               direction[4], direction[7])))),                          # Image Orientation (Patient)
                             ('0028|0030', '\\'.join(map(str, (spacing[0:2])))),                                        # Pixel Spacing
                             ('0028|0100', '16'),                                                                       # Bits Allocated
                             ('0028|0101', '16'),                                                                       # Bits Stored
                             ('0028|0102', '15'),                                                                       # High Bit
                             ('0028|0103', '1'),                                                                        # Pixel Representation
                             ('0028|1052', str(intercept)),                                                             # Rescale Intercept
                             ('0028|1053', str(slope))                                                                  # Rescale Slope
                             ]

        # Use dynamic PET format for multi-component images
        if number_of_components > 1:
            series_tag_values += [('0054|0081', str(number_of_slices)),                                                 # Number of Slices
                                  ('0054|1000', 'DYNAMIC\IMAGE'),                                                       # Series Type
                                  ('0054|0101', str(number_of_components))]                                             # Number of Time Slices

        # For individual image component extraction
        extract_filter = sitk.VectorIndexSelectionCastImageFilter()

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        for i in range(number_of_components):

            if number_of_components > 1:
                extract_filter.SetIndex(i)
                frame = extract_filter.Execute(data)
            else:
                frame = data

            for j in range(number_of_slices):

                image_slice = frame[:, :, j]

                for tag, value in series_tag_values:
                    image_slice.SetMetaData(tag, value)

                image_slice.SetMetaData('0008|0012', time.strftime("%Y%m%d"))                                           # Instance Creation Date
                image_slice.SetMetaData('0008|0013', time.strftime("%H%M%S"))                                           # Instance Creation Time
                image_slice.SetMetaData('0020|0013', str(i*number_of_slices+j))                                         # Instance Number
                image_slice.SetMetaData('0020|0032', '\\'.join(map(str, data.TransformIndexToPhysicalPoint((0, 0, j)))))# Image Position (Patient)

                if number_of_components > 1:
                    image_slice.SetMetaData('0054|1300', str(i))                                                        # Frame ref time

                writer.SetFileName(join(path, str(i*number_of_slices+j) + '.dcm'))
                writer.Execute(image_slice)


class ModelWriter(WriterBase):

    """
    A writer for ModelBase objects.
    """

    def __init__(self):
        super().__init__()

    def write(self, data: ModelBase, path: str):

        """Writes a model.

        Parameters
        ----------
        data : ModelBase
            A model.
        path : str
            The file path.

        """

        pickle.dump(data, open(path, 'wb'))


class SimpleVolumeWriter(WriterBase):

    """
    A writer for medical image volumes stored as single files.
    """

    def __init__(self):
        super().__init__()

    def write(self, data: Image, path: str):

        """Writes an image volume.

        The file format will be determined by SimpleITK.ReadImage based on the file extension if supported.

        Parameters
        ----------
        data : Image
            An image volume.
        path : str
            The file path.

        """

        sitk.WriteImage(data, path)
