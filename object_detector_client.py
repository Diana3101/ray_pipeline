import argparse
import requests
import ray

from patcher import ImagePatcher
from geojson_converter import GeoJsonConverter

import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='Path to directory with input images')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to directory with outputs')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--output_format', required=False, type=str, default='jsonl',
                        help='Format of output file, default jsonl')
    parser.add_argument('--output_label', required=False, type=str, default='object',
                        help='Output field name with results')
    parser.add_argument('--name', required=False, type=str, default='crater',
                        help='Name of the output objects')

    args, unknown = parser.parse_known_args()
    return args


@ray.remote
def send_query(image_array):
    data = {'image_array': image_array.tolist(),
            'is_batching': True}
    response = requests.post("http://127.0.0.1:8000/", json=data)
    return response.json()


def main(args):
    image_patcher = ImagePatcher(input_dir=args.input_dir,
                                 img_size=args.img_size)
    converter = GeoJsonConverter(output_format=args.output_format,
                                 output_dir=args.output_dir,
                                 name=args.name,
                                 output_label=args.output_label)

    generator = image_patcher.batch_generator()

    for batch in generator:
        input_batch = batch
        flatten_image_array_list = [image_array
                                    for image_batch in batch[0]
                                    for image_array in image_batch]

    results = ray.get([send_query.remote(image_array) for image_array in flatten_image_array_list])

    converter.results_to_geotiff(results, input_batch, image_patcher.lens_image_batch)


if __name__ == "__main__":
    args = parse_args()
    main(args)
