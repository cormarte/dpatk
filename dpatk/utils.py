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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: corentin.martens@ulb.be

import SimpleITK
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from SimpleITK import Image


def extract_curves_from_volume(dynamic_volume: SimpleITK.Image, mask: SimpleITK.Image, neighborhood: int=0):

    """Extracts all curves from a 4D volume within a 3D binary mask as a 2D array.

    The curves are averaged within a (2*neighborhood+1) cubic neighborhood around the corresponding voxel, truncated to
    the volume bounds.

    Parameters
    ----------
    dynamic_volume : Image
        A 4D volume.
    mask : Image
        A 3D binary mask. Must have the same spatial dimensions as dynamic_volume.
    neighborhood: int, optional
        The half width of the neighborhood for curve averaging. Default: 0.

    Returns
    -------
    curves : ndarray
        A 2D array containing for all the extracted curves (axis 0) the corresponding samples (axis 1).

    """

    dynamic_volume_array = sitk.GetArrayViewFromImage(dynamic_volume)
    mask_array = sitk.GetArrayViewFromImage(mask)

    assert dynamic_volume_array.ndim == 4 and mask_array.ndim == 3
    assert mask_array.shape[:3] == dynamic_volume_array.shape[:3]

    shape = dynamic_volume_array.shape
    voxels = np.where(mask_array != 0.0)
    slices = [tuple([slice(max(v[i]-neighborhood, 0), min(v[i]+neighborhood+1, shape[i])) for i in range(3)])
              for v in zip(*voxels)]
    curves = np.array([np.mean(dynamic_volume_array[s], axis=(0, 1, 2)) for s in slices])

    return curves


def extract_mean_curve_from_volume(dynamic_volume: SimpleITK.Image, mask: SimpleITK.Image):

    """Extracts the mean curve from a 4D volume within a 3D binary mask as a 1D array.

    Parameters
    ----------
    dynamic_volume : Image
        A 4D volume.
    mask : Image
        A 3D binary mask. Must have the same spatial dimensions as dynamic_volume.

    Returns
    -------
    mean_curve : ndarray
        A 1D array containing the mean curve samples.

    """

    dynamic_volume_array = sitk.GetArrayViewFromImage(dynamic_volume)
    mask_array = sitk.GetArrayViewFromImage(mask)

    assert dynamic_volume_array.ndim == 4 and mask_array.ndim == 3
    assert mask_array.shape[:3] == dynamic_volume_array.shape[:3]

    voxels = np.where(mask_array != 0.0)
    mean_curve = np.mean(dynamic_volume_array[voxels], axis=0)

    return mean_curve


def extract_frames_from_volume(dynamic_volume: SimpleITK.Image, frame_indices: list):

    """Extracts the specified frames from a 4D volume as a new 4D volume.

    Parameters
    ----------
    dynamic_volume : Image
        A 4D volume.
    frame_indices : list
        A list of frame indices to extract.

    Returns
    -------
    output_dynamic_volume : Image
        A new 4D volume containing the extracted frames.

    """

    dynamic_volume_array = sitk.GetArrayViewFromImage(dynamic_volume)

    assert dynamic_volume_array.ndim == 4
    assert all([0 <= i < dynamic_volume_array.shape[3] for i in frame_indices])

    output_dynamic_volume_array = dynamic_volume_array[:, :, :, frame_indices]
    output_dynamic_volume = sitk.GetImageFromArray(output_dynamic_volume_array)
    output_dynamic_volume.CopyInformation(dynamic_volume)

    return output_dynamic_volume


def extract_mean_frame_from_volume(dynamic_volume, frame_indices):

    """Extracts the mean frame for the specified indices from a 4D volume as a 3D volume.

    Parameters
    ----------
    dynamic_volume : Image
        A 4D volume.
    frame_indices : list
        A list of frame indices to average.

    Returns
    -------
    output_volume : Image
        A 3D volume containing the mean frame.

    """

    dynamic_volume_array = sitk.GetArrayViewFromImage(dynamic_volume)

    assert dynamic_volume_array.ndim == 4
    assert all([0 <= i < dynamic_volume_array.shape[3] for i in frame_indices])

    output_volume_array = np.mean(dynamic_volume_array[:, :, :, frame_indices], axis=3)
    output_volume = sitk.GetImageFromArray(output_volume_array)
    output_volume.CopyInformation(dynamic_volume)

    return output_volume


def make_volume_from_curves(curves, mask, default_value=0.0):

    """Recreates a 4D volume from an array of curves originating from a 3D binary mask.

    The curves must be stored in the same order as non-zero voxels of the binary mask are iterated over (x->y->z). This
    typically corresponds to the ordering of the curves returned by dpatk.utils.extract_curves_from_volume.

    Parameters
    ----------
    curves : ndarray
        A 2D array containing for all the provided curves (axis 0) the corresponding samples (axis 1).
    mask : Image
        A 3D binary mask. Must have the same number of non-zero elements as the number number of provided curves.
    default_value : float, optional
        The default sample value for curves outside the binary mask. Default: 0.0.

    Returns
    -------
    dynamic_volume : Image
        A 4D volume containing the mean frame.

    """

    assert curves.ndim == 2

    mask_array = sitk.GetArrayFromImage(mask)
    voxels = np.where(mask_array != 0)

    assert curves.shape[0] == len(voxels[0])

    dynamic_volume_array = default_value*np.ones(mask_array.shape[:3] + (curves.shape[-1],), dtype=np.float64)
    dynamic_volume_array[voxels] = curves
    dynamic_volume = sitk.GetImageFromArray(dynamic_volume_array)
    dynamic_volume.CopyInformation(mask)

    return dynamic_volume
