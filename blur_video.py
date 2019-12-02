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
from . import datasets, decoder, show, transforms, blur_faces, utilities
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

        # Compute radius and center of the blur faces
        radius, number_subjects = blur_faces.Return_Radius (output_path + '.pifpaf.json')
        center_x, center_y = blur_faces.Return_Circle_Center(output_path + '.pifpaf.json')

        # Check if someone is present on the picture
        if radius:

            #For the first picture
            if batch_i == 0:
                # Create reference values
                reference_radius, reference_center_x, reference_center_y = utilities.init_references(number_subjects)

            print('avant')
            print(radius)
            print(center_x)
            print(center_y)
            #print(reference_radius)
            #print(reference_center_x)
            #print(reference_center_y)

            #For all the other pictures
            if batch_i !=0:
                # Check difference of lenght between values and their associated references
                difference_radius, difference_center_x, difference_center_y = utilities.difference_prediction(radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y)

                # Set values and references at the same lenght and in the same order if values are longer than references
                if difference_radius > 0 and difference_center_x > 0 and difference_center_y > 0:
                    radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y = utilities.adjust_length_reference(radius, reference_radius, difference_radius, center_x,  reference_center_x, difference_center_x, center_y, reference_center_y, difference_center_y)
                    print('AA')
                # Set values and references at the same lenght and in the same order if references are longer than values
                if difference_radius < 0 and difference_center_x < 0 and difference_center_y < 0:
                    reference_radius, reference_center_x, reference_center_y = utilities.adjust_length(radius, reference_radius, difference_radius, center_x,  reference_center_x, difference_center_x, center_y, reference_center_y, difference_center_y)
                    print('BB')
                # Order the computed values in an ascending manner to be sure to always compare the reference value and the actual
                # value of the same subject
                if difference_radius == 0:
                    radius, center_x, center_y = utilities.ordered_prediction(radius, reference_radius, center_x, reference_center_x,center_y, reference_center_y)
                    print('CC')

            # Check variation between actual values and values from the previous picture (references).
            variation_radius, variation_center_x, variation_center_y = utilities.variation_prediction(radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y)

            # Adapt values with references if  the variations are significative
            radius, reference_radius = utilities.compare(radius, reference_radius, 6,variation_radius, batch_i)
            center_x, reference_center_x = utilities.compare(center_x, reference_center_x, 5,variation_center_x, batch_i)
            center_y, reference_center_y = utilities.compare(center_y, reference_center_y, 5,variation_center_y, batch_i)

            print('hi')
            print(radius)
            print(center_x)
            print(center_y)

            # Save a new image .blur.png which is the original one with all the faces blurred
            blur_faces.Blur_Face(output_path, output_path + '.pifpaf.json', (361,641,3), number_subjects, radius, center_x, center_y)

        else:
            # Save a new image .blur.png which is the original one if nobody is present on the picture
            shutil.copy(output_path, output_path + ".blur.png")

if __name__ == '__main__':
    main()

#export VIDEO=1.avi  # change to your video file
#mkdir ${VIDEO}.images
#ffmpeg -i ${VIDEO} -qscale:v 2 -vf scale=641:-1 -f image2 ${VIDEO}.images/%05d.jpg
#python3 -m openpifpaf.blur_video --checkpoint resnet152 --glob "${VIDEO}.images/*[05].jpg"
#python3 -m openpifpaf.blur_video --checkpoint shufflenetv2x2 --glob "${VIDEO}.images/*[05].jpg"
#ffmpeg -framerate 24 -pattern_type glob -i ${VIDEO}.images/'*.jpg.blur.png' -vf scale=640:-2 -c:v libx264 -pix_fmt yuv420p ${VIDEO}.pose.mp4
