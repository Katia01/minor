"""
Created on November 2019

@author: Katia Schalk

Predict poses for all the images constituting the video.

To run the code execute the following command on the terminal:
    export VIDEO=your_video_filename.avi
    mkdir ${VIDEO}.images
    ffmpeg -i ${VIDEO} -qscale:v 2 -vf scale=641:-1 -f image2 ${VIDEO}.images/%05d.jpg
    python3 -m openpifpaf.posture_estimation --checkpoint resnet152 --glob "${VIDEO}.images/*[05].jpg"
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import PIL
import torch
import statistics

from .network import nets
from . import datasets, decoder, show, transforms, return_coordinates, posture_image, utilities, identify_patient, from_pixel_to_meter, save_coordinates, extract_z_coordinates, posture_image

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json'],
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--line-width', default=6, type=int,
                        help='line width for skeleton')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # Organize the list in ascending order ensuring to predict from the first recording image to the last one
    args.images = utilities.sorted_aphanumeric(args.images)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def bbox_from_keypoints(kps):
    m = kps[:, 2] > 0
    if not np.any(m):
        return [0, 0, 0, 0]

    x, y = np.min(kps[:, 0][m]), np.min(kps[:, 1][m])
    w, h = np.max(kps[:, 0][m]) - x, np.max(kps[:, 1][m]) - y
    return [x, y, w, h]


def main():
    args = cli()

    # load model
    model_cpu, _ = nets.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.head_names = model_cpu.head_names
        model.head_strides = model_cpu.head_strides
    processor = decoder.factory_from_args(args, model, args.device)

    # data
    preprocess = None
    if args.long_edge:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])
    data = datasets.ImageList(args.images, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    keypoint_painter = show.KeypointPainter()
    skeleton_painter = show.KeypointPainter(
        color_connections=True,
        markersize=args.line_width - 5,
        linewidth=args.line_width,
    )

    # Since prediction starts for the image 5 (first image is 1), the z coordinates starts for array 4 (first array is 0)
    array_number = 4
    time = 0.0

    # Create a .csv file where the xyz coordinates will be saved  at the end
    save_coordinates.Create_csv_File('xyz_coordinates.csv', 'KNE_angle.csv')

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        fields_batch = processor.fields(image_tensors_batch)
        pred_batch = processor.annotations_batch(fields_batch, debug_images=image_tensors_batch)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            if args.output_directory is None:
                output_path = meta['file_name']
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, file_name)
            LOG.info('batch %d: %s to %s', batch_i, meta['file_name'], output_path)

            # load the original image if necessary
            cpu_image = None
            if args.debug or \
               'keypoints' in args.output_types or \
               'skeleton' in args.output_types:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            processor.set_cpu_image(cpu_image, None)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            if 'json' in args.output_types:
                with open(output_path + '.pifpaf.json', 'w') as f:
                    json.dump([
                        {
                            'keypoints': np.around(ann.data, 1).reshape(-1).tolist(),
                            'bbox': np.around(bbox_from_keypoints(ann.data), 1).tolist(),
                            'score': round(ann.score(), 3),
                        }
                        for ann in pred
                    ], f)

            if 'keypoints' in args.output_types:
                with show.image_canvas(cpu_image,
                                       output_path + '.keypoints.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    keypoint_painter.annotations(ax, pred)

            if 'skeleton' in args.output_types:
                with show.image_canvas(cpu_image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.annotations(ax, pred)

        # Set this value equal to the distance between the subject and the camera + 1 meter
        z_limit = 4500

        # Return the xy reference patient return_coordinates
        xy_reference_patient = identify_patient.Identify_Patient_Coordinates(output_path)

        # Read the given .json coordinates file and Return the left and right xy coordinates of all subjects present on the picture
        REAR_x, REAR_y, LEAR_x, LEAR_y = return_coordinates.Return_Ears_Coordinates(output_path + '.pifpaf.json')
        RSHO_x, RSHO_y, LSHO_x, LSHO_y = return_coordinates.Return_Shoulders_Coordinates(output_path + '.pifpaf.json')
        RTHI_x, RTHI_y, LTHI_x, LTHI_y = return_coordinates.Return_Hips_Coordinates(output_path + '.pifpaf.json')
        RKNE_x, RKNE_y, LKNE_x, LKNE_y = return_coordinates.Return_Knees_Coordinates(output_path + '.pifpaf.json')
        RANK_x, RANK_y, LANK_x, LANK_y = return_coordinates.Return_Ankles_Coordinates(output_path + '.pifpaf.json')

        # Return the vector index corresponding to the patient
        index = identify_patient.Determine_Patient_index(RSHO_x, RSHO_y, LSHO_x, LSHO_y, xy_reference_patient)

        # Extract the xy coordinates of the patient
        REAR_patient_xy, LEAR_patient_xy = identify_patient.Select_Patient_Coordinates(REAR_x, REAR_y, LEAR_x, LEAR_y, index)
        RSHO_patient_xy, LSHO_patient_xy = identify_patient.Select_Patient_Coordinates(RSHO_x, RSHO_y, LSHO_x, LSHO_y, index)
        RTHI_patient_xy, LTHI_patient_xy = identify_patient.Select_Patient_Coordinates(RTHI_x, RTHI_y, LTHI_x, LTHI_y, index)
        RKNE_patient_xy, LKNE_patient_xy = identify_patient.Select_Patient_Coordinates(RKNE_x, RKNE_y, LKNE_x, LKNE_y, index)
        RANK_patient_xy, LANK_patient_xy = identify_patient.Select_Patient_Coordinates(RANK_x, RANK_y, LANK_x, LANK_y, index)

        # Return the xyz coordinates of the patient
        if REAR_patient_xy[0] != 0 and REAR_patient_xy[1] != 0 and LEAR_patient_xy[0] != 0 and LEAR_patient_xy[1] != 0:
            REAR_patient_xyz, LEAR_patient_xyz = extract_z_coordinates.Return_xyz_Coordinates(REAR_patient, LEAR_patient, array_number, z_limit)
            REAR_patient_xyz = from_pixel_to_meter.Convert_Pixel_To_Meter(REAR_patient_xyz)
            LEAR_patient_xyz = from_pixel_to_meter.Convert_Pixel_To_Meter(LEAR_patient_xyz)
        else:
            REAR_patient_xyz = [0, 0, 0]
            LEAR_patient_xyz = [0, 0, 0]

        if RSHO_patient_xy[0] != 0 and RSHO_patient_xy[1] != 0 and LSHO_patient_xy[0] != 0 and LSHO_patient_xy[1] != 0:
            RSHO_patient_xyz, LSHO_patient_xyz = extract_z_coordinates.Return_xyz_Coordinates(RSHO_patient_xy, LSHO_patient_xy, array_number, z_limit)
            RSHO_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(RSHO_patient_xyz)
            LSHO_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(LSHO_patient_xyz)
        else:
            RSHO_patient_xyz = [0, 0, 0]
            LSHO_patient_xyz = [0, 0, 0]

        if RTHI_patient_xy[0] != 0 and RTHI_patient_xy[1] != 0 and LTHI_patient_xy[0] != 0 and LTHI_patient_xy[1] != 0:
            RTHI_patient_xyz, LTHI_patient_xyz = extract_z_coordinates.Return_xyz_Coordinates(RTHI_patient_xy, LTHI_patient_xy, array_number, z_limit)
            RTHI_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(RTHI_patient_xyz)
            LTHI_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(LTHI_patient_xyz)
        else:
            RTHI_patient_xyz = [0, 0, 0]
            LTHI_patient_xyz = [0, 0, 0]

        if RKNE_patient_xy[0] != 0 and RKNE_patient_xy[1] != 0 and LKNE_patient_xy[0] != 0 and LKNE_patient_xy[1] != 0:
            RKNE_patient_xyz, LKNE_patient_xyz = extract_z_coordinates.Return_xyz_Coordinates(RKNE_patient_xy, LKNE_patient_xy, array_number, z_limit)
            RKNE_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(RKNE_patient_xyz)
            LKNE_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(LKNE_patient_xyz)
        else:
            RKNE_patient_xyz = [0, 0, 0]
            LKNE_patient_xyz = [0, 0, 0]

        if RANK_patient_xy[0] != 0 and RANK_patient_xy[1] != 0 and LANK_patient_xy[0] != 0 and LANK_patient_xy[1] != 0:
            RANK_patient_xyz, LANK_patient_xyz = extract_z_coordinates.Return_xyz_Coordinates(RANK_patient_xy, LANK_patient_xy, array_number, z_limit)
            RANK_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(RANK_patient_xyz)
            LANK_patient_xyz  = from_pixel_to_meter.Convert_Pixel_To_Meter(LANK_patient_xyz)
        else:
            RANK_patient_xyz = [0, 0, 0]
            LANK_patient_xyz = [0, 0, 0]

        save_coordinates.Save_Coordinates_csv(REAR_patient_xyz, LEAR_patient_xyz, RSHO_patient_xyz, LSHO_patient_xyz, RTHI_patient_xyz, LTHI_patient_xyz, RKNE_patient_xyz, LKNE_patient_xyz, RANK_patient_xyz, LANK_patient_xyz, 'xyz_coordinates.csv')

        # Compute right and left knee angle
        if RTHI_patient_xy[0] != 0 and RTHI_patient_xy[1] != 0 and RKNE_patient_xy[0] != 0 and RKNE_patient_xy[1] != 0 and RANK_patient_xy[0] != 0 and RANK_patient_xy[1] != 0:
            RKNE_angle = posture_image.Compute_Knee_Angle (RTHI_patient_xyz, RKNE_patient_xyz, RANK_patient_xyz)
        else:
            RKNE_angle = 0.0

        if LTHI_patient_xy[0] != 0 and LTHI_patient_xy[1] != 0 and LKNE_patient_xy[0] != 0 and LKNE_patient_xy[1] != 0 and LANK_patient_xy[0] != 0 and LANK_patient_xy[1] != 0:
            LKNE_angle = posture_image.Compute_Knee_Angle (LTHI_patient_xyz, LKNE_patient_xyz, LANK_patient_xyz)
        else:
            LKNE_angle = 0.0

        save_coordinates.Save_Angle_csv(RKNE_angle, LKNE_angle, 'KNE_angle.csv', time)

        # Switch to the z array corresponding to the new image
        array_number = array_number + 5
        time = time + 0.10

if __name__ == '__main__':
    main()
