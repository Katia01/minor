"""
Created on September 2019

@author: Katia Schalk

Blur faces on the entire video.

To run the code execute the following command on the terminal:
    export VIDEO=your_video_filename.avi
    mkdir ${VIDEO}.images
    ffmpeg -i ${VIDEO} -qscale:v 2 -vf scale=641:-1 -f image2 ${VIDEO}.images/%05d.jpg
    python3 -m openpifpaf.blur_video --checkpoint resnet152 --glob "${VIDEO}.images/*[05].jpg"
    ffmpeg -framerate 24 -pattern_type glob -i ${VIDEO}.images/'*.jpg.blur.png' -vf scale=640:-2 -c:v libx264 -pix_fmt yuv420p ${VIDEO}.pose.mp4
"""
import argparse
import glob
import json
import logging
import os
import re

import numpy as np
import PIL
import torch

from .network import nets
from . import datasets, decoder, show, transforms, blur_faces, utilities, return_coordinates, field_of_view, extract_z_coordinates
import shutil

def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.blur_video',
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
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
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
    processor = decoder.factory_from_args(args, model)

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
    keypoint_painter = show.KeypointPainter(show_box=False)
    skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=6)


        #Hugo2
    array_number = 47
    array_name = "/Users/KatiaSchalk/Desktop/openpifpaf/depth_values_Hugo2/array_"

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        image_tensors_batch_gpu = image_tensors_batch.to(args.device, non_blocking=True)
        fields_batch = processor.fields(image_tensors_batch_gpu)
        pred_batch = processor.annotations_batch(fields_batch, debug_images=image_tensors_batch)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            if args.output_directory is None:
                output_path = meta['file_name']
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, file_name)

            logging.info('batch %d: %s to %s', batch_i, meta['file_name'], output_path)

            # load the original image if necessary
            cpu_image = None
            if args.debug or \
               'keypoints' in args.output_types or \
               'skeleton' in args.output_types:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            processor.set_cpu_image(cpu_image, None)
            keypoint_sets, scores = processor.keypoint_sets_from_annotations(pred)
            if preprocess is not None:
                keypoint_sets = preprocess.keypoint_sets_inverse(keypoint_sets, meta)

            if 'json' in args.output_types:
                with open(output_path + '.pifpaf.json', 'w') as f:
                    json.dump([
                        {
                            'keypoints': np.around(kps, 1).reshape(-1).tolist(),
                            'bbox': bbox_from_keypoints(kps),
                        }
                        for kps in keypoint_sets
                    ], f)

            if 'keypoints' in args.output_types:
                with show.image_canvas(cpu_image,
                                       output_path + '.keypoints.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    keypoint_painter.keypoints(ax, keypoint_sets)

            if 'skeleton' in args.output_types:
                with show.image_canvas(cpu_image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

        # Set this value equal to the distance between the subject and the camera + 1 meter
        z_limit = 8000

        # Read the given .json coordinates file and Return the left and right xy coordinates of all subjects present on the picture
        Nose_x, Nose_y = return_coordinates.Return_Nose_Coordinates(output_path + '.pifpaf.json')
        REAR_x, REAR_y, LEAR_x, LEAR_y = return_coordinates.Return_Ears_Coordinates(output_path + '.pifpaf.json')
        RSHO_x, RSHO_y, LSHO_x, LSHO_y = return_coordinates.Return_Shoulders_Coordinates(output_path + '.pifpaf.json')

        number_subjects = len(Nose_x)

        true = os.path.isfile(array_name + str(array_number) + ".npy" )

        array_radius = []
        array_center_x = []
        array_center_y = []

        if number_subjects != 0:

            for subject in range(number_subjects):
                Nose_xy = []
                Nose_xy.append(Nose_x[subject])
                Nose_xy.append(Nose_y[subject])

                REAR_xy = []
                REAR_xy.append(REAR_x[subject])
                REAR_xy.append(REAR_y[subject])

                LEAR_xy = []
                LEAR_xy.append(LEAR_x[subject])
                LEAR_xy.append(LEAR_y[subject])

                RSHO_xy = []
                RSHO_xy.append(RSHO_x[subject])
                RSHO_xy.append(RSHO_y[subject])

                LSHO_xy = []
                LSHO_xy.append(LSHO_x[subject])
                LSHO_xy.append(LSHO_y[subject])

                Nose_xy, Nose_xy = field_of_view.Correct_Shift(Nose_xy, Nose_xy )
                REAR_xy, LEAR_xy = field_of_view.Correct_Shift(REAR_xy, LEAR_xy )
                RSHO_xy, LSHO_xy = field_of_view.Correct_Shift(RSHO_xy, LSHO_xy)

                # Return the xyz coordinates of the subject
                if true and Nose_xy[0] != 0 and Nose_xy[1] != 0:
                    Nose_xyz = extract_z_coordinates.Return_Center_xyz_Coordinates(Nose_xy, array_number, z_limit, array_name)
                else:
                    Nose_xyz = [0, 0, 0]

                if true and REAR_xy[0] != 0 and REAR_xy[1] != 0 and LEAR_xy[0] != 0 and LEAR_xy[1] != 0:
                    REAR_xyz, LEAR_xyz = extract_z_coordinates.Return_xyz_Coordinates(REAR_xy, LEAR_xy, array_number, z_limit, array_name)
                else:
                    REAR_xyz = [0, 0, 0]
                    LEAR_xyz = [0, 0, 0]

                if true and RSHO_xy[0] != 0 and RSHO_xy[1] != 0 and LSHO_xy[0] != 0 and LSHO_xy[1] != 0:
                    RSHO_xyz, LSHO_xyz = extract_z_coordinates.Return_xyz_Coordinates(RSHO_xy, LSHO_xy, array_number, z_limit, array_name)
                else:
                    RSHO_xyz = [0, 0, 0]
                    LSHO_xyz = [0, 0, 0]

                # Compute radius and center of the blur faces
                if Nose_xyz !=0:
                    radius = blur_faces.Return_Radius(Nose_xyz)
                    center_x, center_y = blur_faces.Return_Circle_Center(Nose_x[subject], Nose_y[subject])

                else:
                    if REAR_xyz !=0 and LEAR_xyz == 0:
                        radius = blur_faces.Return_Radius(REAR_xyz)
                        center_x, center_y = blur_faces.Return_Circle_Center(REAR_x[subject], REAR_y[subject])
                    else:
                        if LEAR_xyz !=0 and REAR_xyz ==0:
                            radius = blur_faces.Return_Radius(LEAR_xyz)
                            center_x, center_y = blur_faces.Return_Circle_Center(LEAR_x[subject], LEAR_y[subject])
                        else:
                            if LEAR_xyz !=0 and REAR_xyz !=0:
                                CEAR_xyz = []
                                CEAR_xyz.append((REAR_xyz[0]+ LEAR_xyz[0])/2);
                                CEAR_xyz.append((REAR_xyz[1]+ LEAR_xyz[1])/2);
                                CEAR_xyz.append((REAR_xyz[2]+ LEAR_xyz[2])/2);

                                radius = blur_faces.Return_Radius(CEAR_xyz)
                                center_x, center_y = blur_faces.Return_Circle_Center((LEAR_x[subject]+LEAR_x[subject])/2 , (LEAR_y[subject]+LEAR_y[subject])/2)

                array_radius.append(radius)
                array_center_x.append(center_x)
                array_center_y.append(center_y)

            # Save a new image .blur.png which is the original one with all the faces blurred
            blur_faces.Blur_Face(output_path, output_path + '.pifpaf.json', (481,641,3), number_subjects, array_radius, array_center_x, array_center_y)
        else:
            # Save a new image .blur.png which is the original one if nobody is present on the picture
            shutil.copy(output_path, output_path + ".blur.png")

        # Switch to the z array corresponding to the new image
        array_number = array_number + 5

if __name__ == '__main__':
    main()

#export VIDEO=1.avi  # change to your video file
#mkdir ${VIDEO}.images
#ffmpeg -i ${VIDEO} -qscale:v 2 -vf scale=641:-1 -f image2 ${VIDEO}.images/%05d.jpg
#python3 -m openpifpaf.blur_video --checkpoint resnet152 --glob "${VIDEO}.images/*[05].jpg"
#python3 -m openpifpaf.blur_video --checkpoint shufflenetv2x2 --glob "${VIDEO}.images/*[05].jpg"
#ffmpeg -framerate 24 -pattern_type glob -i ${VIDEO}.images/'*.jpg.blur.png' -vf scale=640:-2 -c:v libx264 -pix_fmt yuv420p ${VIDEO}.pose.mp4
