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

import numpy as np
import os
import SimpleITK as sitk
from dpatk.utils import extract_frames_from_volume, extract_mean_frame_from_volume
from scipy.signal import correlate
from SimpleITK import Image, VersorRigid3DTransform


def compute_curve_delay(fixed_curve, moving_curve):

    """Computes the delay of a moving curve wrt a fixed (reference) curve by cross-correlation maximization.

    Both curves must have the same sampling and number of samples.

    Parameters
    ----------
    fixed_curve : ndarray
        A 1D array containing the samples of the fixed (reference) curve.
    moving_curve : ndarray
        A 1D array containing the samples of the moving (registered) curve. Must have the same sampling and number of
        samples as fixed_curve.

    Returns
    -------
    delay : int
        The estimated signed delay of the moving curve wrt to the fixed curve in number of samples.
    """

    assert fixed_curve.ndim == 1
    assert moving_curve.ndim == 1
    assert len(fixed_curve) == len(moving_curve)

    cross_correlation = correlate(fixed_curve, moving_curve, mode='full', method='direct')
    argmax = np.argmax(cross_correlation)
    delay = min(argmax%fixed_curve.shape[0], argmax-fixed_curve.shape[0])

    return delay


def register_curve(fixed_curve, moving_curve):

    """Registers a moving curve to a fixed (reference) curve by cross-correlation maximization.

    Both curves must have the same sampling and number of samples.

    Parameters
    ----------
    fixed_curve : ndarray
        A 1D array containing the samples of the fixed (reference) curve.
    moving_curve : ndarray
        A 1D array containing the samples of the moving (registered) curve. Must have the same sampling and number of
        samples as fixed_curve.

    Returns
    -------
    registered_curve : ndarray
        A 1D array containing the samples of the registered curve.
    delay : int
        The estimated signed delay of the moving curve wrt to the fixed curve in number of samples.
    """

    delay = compute_curve_delay(fixed_curve, moving_curve)
    registered_curve = shift_curve(moving_curve, delay)

    return registered_curve, delay


def register_dynamic_volume(dynamic_volume, block_index_ranges, reference_block_index=-1, nb_of_bins=50,
                            use_memmap=False, delete_original_volume=False):

    """Rigidly registers all frames of a 4D volume by mutual information maximization.

    The frames are grouped into blocks and blockwise averaged to increase their SNR. The blocks are then registered to a
    user-specified block of reference. Individual frame transforms are finally linearly interpolated from the previously
    estimated block transforms.

    Parameters
    ----------
    dynamic_volume : Image
        A 4D volume.
    block_index_ranges : list of range
        A list of ranges specifying the frame indices of each block.
    reference_block_index: int, optional
        The index of the block used as reference for block registration. Default: -1.
    nb_of_bins: int, optional
        The number of bins for mutual information computation. Must be > 0. Default: 50.
    use_memmap: bool, optional
        Specifies whether to use a memmap for the temporary registered array. Set it to true in case of memory errors.
        Default: False.
    delete_original_volume: bool, optional
        Specifies whether to delete dynamic_volume before allocating registered_dynamic_volume. Set it to true in case
        of memory errors. Default: False.

    Returns
    -------
    registered_dynamic_volume : Image
        The registered 4D volume.

    """

    assert abs(reference_block_index) < len(block_index_ranges)
    assert nb_of_bins > 0

    dynamic_volume_array = sitk.GetArrayViewFromImage(dynamic_volume)
    direction = dynamic_volume.GetDirection()
    origin = dynamic_volume.GetOrigin()
    spacing = dynamic_volume.GetSpacing()

    if use_memmap:
        registered_dynamic_volume_array = np.lib.format.open_memmap('./tmp.npy', 'w+', shape=dynamic_volume_array.shape,
                                                                    dtype=np.float64)
    else:
        registered_dynamic_volume_array = np.zeros_like(dynamic_volume_array)

    original_blocks_array = np.zeros(dynamic_volume_array.shape[:-1] + (len(block_index_ranges),), dtype=np.float64)
    registered_blocks_array = np.zeros_like(original_blocks_array)

    fixed_frame = extract_mean_frame_from_volume(dynamic_volume, list(block_index_ranges[reference_block_index]))

    fixed_parameters = None
    parameters = []

    for i in range(len(block_index_ranges)):

        moving_frame = extract_mean_frame_from_volume(dynamic_volume, list(block_index_ranges[i]))
        moving_array = sitk.GetArrayViewFromImage(moving_frame)

        if i != reference_block_index:

            print(f'Registering block {i}.')
            transform = versor_rigid_register_volume(fixed_frame, moving_frame, nb_of_bins=nb_of_bins)
            fixed_parameters = transform.GetFixedParameters()
            parameters.append(transform.GetParameters())
            registered_frame = versor_rigid_transform_volume(moving_frame, transform.GetFixedParameters(),
                                                             transform.GetParameters())
            registered_array = sitk.GetArrayViewFromImage(registered_frame)

        else:

            print(f'Copying reference block {i}.')
            parameters.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            registered_array = moving_array

        registered_blocks_array[:, :, :, i] = registered_array
        original_blocks_array[:, :, :, i] = moving_array

    parameters = np.array(parameters)
    block_centers = np.array([0.5*(list(b)[0]+list(b)[-1]) for b in block_index_ranges])

    for i in range(dynamic_volume_array.shape[-1]):

        current_parameters = [np.interp(i, block_centers, parameters[:, j]) for j in range(parameters.shape[1])]
        current_frame = extract_frames_from_volume(dynamic_volume, [i])

        print(f'Transforming frame {i} with parameters {current_parameters}.')
        registered_frame = versor_rigid_transform_volume(current_frame, fixed_parameters, current_parameters)
        registered_dynamic_volume_array[:, :, :, i] = sitk.GetArrayFromImage(registered_frame)

    if use_memmap:
        registered_dynamic_volume_array.flush()

    if delete_original_volume:
        del dynamic_volume

    registered_dynamic_volume = sitk.GetImageFromArray(registered_dynamic_volume_array)
    registered_dynamic_volume.SetDirection(direction)
    registered_dynamic_volume.SetOrigin(origin)
    registered_dynamic_volume.SetSpacing(spacing)

    registered_blocks = sitk.GetImageFromArray(registered_blocks_array)
    registered_blocks.SetDirection(direction)
    registered_blocks.SetOrigin(origin)
    registered_blocks.SetSpacing(spacing)

    original_blocks = sitk.GetImageFromArray(original_blocks_array)
    original_blocks.SetDirection(direction)
    original_blocks.SetOrigin(origin)
    original_blocks.SetSpacing(spacing)

    if use_memmap:
        os.remove('./tmp.npy')
    else:
        del registered_dynamic_volume_array

    return registered_dynamic_volume, original_blocks, registered_blocks, parameters


def shift_curve(curve, delay):

    """Shifts a curve by a given signed delay in number of samples.

    Constant padding with the last known sample value is used for unknown samples.

    Parameters
    ----------
    curve : ndarray
        A 1D array containing the samples of the curve.
    delay : int
        The signed delay in number of samples.

    Returns
    -------
    shifted_curve : ndarray
        A 1D array containing the samples of the shifted curve.
    """

    shifted_curve = np.zeros(curve.shape)

    if delay > 0:
        shifted_curve[0:delay] = curve[0]
        shifted_curve[delay:] = curve[0:-delay]
    elif delay < 0:
        shifted_curve[0:delay] = curve[-delay:]
        shifted_curve[delay:] = curve[-1]
    else:
        shifted_curve[:] = curve[:]

    return shifted_curve


def versor_rigid_register_volume(fixed_volume, moving_volume, nb_of_bins=50):

    """Rigidly registers a moving 3D volume to a fixed 3D volume by mutual information maximization using versors.

    Parameters
    ----------
    fixed_volume : Image
        The fixed (reference) 3D volume.
    moving_volume : Image
        The moving (registered) 3D volume.
    nb_of_bins: int, optional
        The number of bins for mutual information computation. Must be > 0. Default: 50.

    Returns
    -------
    transform : VersorRigid3DTransform
        The estimated transform.

    """

    assert nb_of_bins > 0

    # Registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=nb_of_bins)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    # Optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=8.0, minStep=0.00001,
                                                                 numberOfIterations=500, relaxationFactor=0.8,
                                                                 gradientMagnitudeTolerance=0.00001)
    registration_method.SetOptimizerScalesFromPhysicalShift(smallParameterVariation=0.01)

    # Multi-level parameters
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed_volume, moving_volume, sitk.VersorRigid3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Run
    transform = registration_method.Execute(fixed_volume, moving_volume)

    return transform


def versor_rigid_transform_volume(volume, fixed_parameters, parameters):

    """Applies a versor rigid transform to a 3D volume.

    Parameters
    ----------
    volume : Image
        A 3D volume.
    fixed_parameters : list of float
        The fixed parameters of the transform.
    parameters: list of float
        The estimated parameters of the transform.

    Returns
    -------
    transformed_volume : Image
        The transformed 3D volume.

    """

    transform = sitk.VersorRigid3DTransform()
    transform.SetFixedParameters(fixed_parameters)
    transform.SetParameters(parameters)

    transformed_volume = sitk.Resample(volume, transform, sitk.sitkLinear, 0.0)

    return transformed_volume
