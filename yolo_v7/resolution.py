import argparse
import rasterio


def get_resolution(args):
    img_path = args.img_path
    from pyproj import CRS, Transformer
    import pymap3d as pm

    with rasterio.open(img_path, 'r') as img:
        bounds = img.bounds
        shape = img.shape
        csr = img.crs

        proj = Transformer.from_crs(csr, 4326, always_xy=True)
        lon0, lat0 = proj.transform(bounds.left, bounds.bottom)
        lon1, lat1 = proj.transform(bounds.right, bounds.top)

        # lat, long
        e, n, u = pm.geodetic2enu(lat1, lon1, 0, lat0, lon0, 0)

        x_res = e / shape[1]
        y_res = n / shape[0]

        print(x_res, y_res)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True, type=str, help='Path to input image')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_resolution(args)
