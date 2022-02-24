import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from keras.utils.generic_utils import Progbar
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont


def detect_sequence_imgs(yolo, list_images, output_dir, save_img=False):

    # Prepare directory to save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load txt files with the paths of all the images
    with open(list_images) as f:
        lines = f.readlines()

    # Prepare progress bar
    steps = len(lines)
    progbar = Progbar(target=steps)

    # Iterate over each of the images
    for i in range(0, steps):

        # Load image
        lines[i] = lines[i].replace("\n", "")
        if not lines[i].endswith(('.jpg')):
            lines[i] += '.jpg'
        try:
            img = Image.open(lines[i])
        except:
            print('Open Error! Try again!')
            continue
        else:

            # Make predictions
            predictions = yolo.detection_results(img, lines[i])

            # Create dir if not exists
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Save the results on a txt (one txt per image)
            results = open(os.path.join(output_dir, lines[i].replace("/", "__").replace('.jpg', ".txt")), 'w')

            if save_img:
                font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                          size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
                thickness = (img.size[0] + img.size[1]) // 300
                draw = ImageDraw.Draw(img)

            for j in range(0, predictions.shape[0]):

                if predictions.shape[1] > 1:
                    label = predictions[j, 0].split()

                    left = predictions[j, 1]
                    top = predictions[j, 2]
                    right = predictions[j, 3]
                    bottom = predictions[j, 4]

                    print('\tClass = {}, Confidence = {}, Xmin = {}, Ymin = {}, Xmax = {}, Ymax = {}'.format(label[0],
                          label[1], left, top, right, bottom))
                    results.write(predictions[j, 0] + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n')

                    if save_img:
                        left = int(left)
                        top = int(top)
                        right = int(right)
                        bottom = int(bottom)

                        label_size = draw.textsize(predictions[j, 0], font)

                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])

                        # My kingdom for a good redistributable image drawing library.
                        for k in range(thickness):
                            draw.rectangle([left + k, top + k, right - k, bottom - k], outline=(0, 255, 0))

                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0, 255, 0))
                        draw.text(text_origin, predictions[j, 0], fill=(0, 0, 0), font=font)

                else:
                    results.write(predictions[j, 0])

            results.close()

            # Insert the results on the jpg
            if save_img:
                img.save(os.path.join(output_dir, lines[i].replace("/", "__").replace("jpg", "png")), 'PNG')
                del draw

        # Update progress bar
        progbar.update(i+1), print('\t')

    yolo.close_session()


def detect_img(yolo, image_path, output_dir='', gt='', save_img=False):

    # Load image
    try:
        image= Image.open(image_path)
    except:
        print('Open Error! Try again!')
    else:

        # Make predictions
        img_pred, predictions = yolo.detect_image(image)

        # If gt is set, insert the gts on the image
        if len(gt) > 0:

            # Define font of the text of the labels and the thickness of the boxes
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            # Load gt
            with open(gt) as g:
                lines = g.readlines()

            # Iterate over each bounding box
            for i in range(0, len(lines)):

                # Prepare to draw on the image
                base = img_pred.convert('RGBA')
                img_alpha = Image.new('RGBA', base.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(img_alpha, 'RGBA')

                # Get label
                lines[i] = lines[i].replace("\n", "")
                line = lines[i].split()
                label = line[0]

                # Get the coordinates
                left = float(line[1])
                top = float(line[2])
                right = float(line[3])
                bottom = float(line[4])

                # Prepare box for the label
                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # Iterate to draw the box more thick
                for k in range(thickness):
                    draw.rectangle([left + k, top + k, right - k, bottom - k], outline=(255, 0, 0, 255))


                # Draw the box of the labels
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 0, 0, 64))

                # Insert the text of the labels
                draw.text(text_origin, label, fill=(0, 0, 0, 255), font=font)
                img_pred = Image.alpha_composite(base, img_alpha)
                del draw

        # If output_dir is set, save the results on a txt and a jpg
        if len(output_dir) > 0:

            # Create dir if not exists
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Insert the results on the txt
            results = open(output_dir + "/prediction.txt", 'w')
            for j in range(0, predictions.shape[0]):
                if predictions.shape[1] > 1:
                    results.write(predictions[j, 0] + ' ' + predictions[j, 1] + ' ' + predictions[j, 2] + ' '
                        + predictions[j, 3] + ' ' + predictions[j, 4] + '\n')
                else:
                    results.write(predictions[j, 0])
            results.close()

            # Insert the results on the jpg
            if save_img:
                img_pred.save(output_dir + "/prediction.png")

        # Plot the result
        plt.imshow(np.array(img_pred)), plt.show()

    yolo.close_session()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


FLAGS = None


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str, dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--image_size', type=int, dest='image_size',
        help='Size of the processed image(s), same for width and height,'
             'default  ({}, {})'.format(YOLO.get_defaults("model_image_size")[0],
                                        YOLO.get_defaults("model_image_size")[1])
    )

    parser.add_argument(
        '--score', type=float, dest='score', default=0.3,
        help='confidence score threshold, default ' + str(YOLO.get_defaults("score"))
    )

    parser.add_argument(
        '--iou', type=float, dest='iou', default=0.5,
        help='IoU threshold, default ' + str(YOLO.get_defaults("iou"))
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False, default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--save_img", nargs='?', type=str2bool, default=False,
        help="Save image with predictions or not"
    )

    # Only for single detections (single_image=True)
    parser.add_argument(
        "--gt", nargs='?', type=str, default="",
        help="Path to txt file with the ground truth of the given image"
    )

    # Required for multiple detections (single_image=False)
    parser.add_argument(
        "--list_images", nargs='?', type=str, default="/home/nsp/Desktop/TFM_Natalia/PIROPO/Test/lista_img_test.txt",
        help="Path to txt file listing the path of each of the images"
    )

    # Required for multiple detections (single_image=False)
    parser.add_argument(
        "--single_image", nargs='?', type=str, default="",
        help="If you wish to detect one single image, indicate here its path"
    )

    parser.add_argument(
        "--output_dir", nargs='?', type=str, default="/home/nsp/Desktop/TFM_Natalia/keras-yolo3-master/input/detection-results-trial0",
        help="Path where a set of txt files (one per image) will be generated"
    )

    FLAGS = parser.parse_args()

    if len(FLAGS.single_image) > 0:
        detect_img(YOLO(**vars(FLAGS)), image_path=FLAGS.single_image, output_dir=FLAGS.output_dir, gt=FLAGS.gt, save_img=FLAGS.save_img)
    else:
        detect_sequence_imgs(YOLO(**vars(FLAGS)), list_images=FLAGS.list_images, output_dir=FLAGS.output_dir, save_img=FLAGS.save_img)

