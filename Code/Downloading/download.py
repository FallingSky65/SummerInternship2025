# -----------------------------------------------------------------------------
# File          : download.py
# Description   : Downloads TROPOMI L2 SO2 data from 2021 to 2024
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
import earthaccess
from netCDF4 import Dataset
from datetime import datetime
from sys import exit
import os
from pathlib import Path
import numpy as np

# parameters
ATTEMPTS   =  5 # number of attempts before giving up operation
BATCH_SIZE = 50 # batch size for files to download and process

# paths
PROJ_ROOT = os.getenv('PROJ_ROOT', '/explore/nobackup/people/dgli/Clone')
DATA_PATH = f'{PROJ_ROOT}/Data/TROPOMI/S5P_L2__SO2____HiR'
TEMP_PATH = f'{DATA_PATH}/temp'


def batch(iterable, n=BATCH_SIZE):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


# returns exit code
def download():
    # clean up files from possible previous download attempt
    if Path(TEMP_PATH).exists():
        for f in Path(TEMP_PATH).iterdir():
            if f.is_file() and f.suffix == '.nc':
                os.remove(f)

    # log in to earthaccess
    print('logging in to earthaccess...')
    auth = earthaccess.login(persist=True)
    if not auth.authenticated:
        print('failed to login.')
        return 1
    print('logged in')

    # query for data
    success = False
    print('querying data')
    for i in range(ATTEMPTS):
        try:
            results = earthaccess.search_data(
                short_name="S5P_L2__SO2____HiR",
                temporal=('2021-01-01 00:00', '2025-01-01 00:00')
            )
        except:
            print('failed query, retrying...')
        else:
            success = True
            break

    if not success:
        print('failed query attempts, giving up.')
        return 1

    print(f'{len(results)} granules found')

    open_fails = 0
    for result_batch in batch(results):
        # download results in batches
        print('downloading batch...')
        success = False
        for i in range(ATTEMPTS):
            try:
                earthaccess.download(result_batch, TEMP_PATH)
            except:
                print('failed download, retrying...')
            else:
                success = True
                break
        if not success:
            print('failed download nc attempts, giving up.')
            return 1
        print('download success, saving...')

        # save only wanted variables
        for f in Path(TEMP_PATH).iterdir():
            if f.is_file() and f.suffix == '.nc':
                print(f'processing {f.name}...')

                # save files by date
                fmt = '%Y%m%dT%H%M%S'
                start = datetime.strptime(f.name[20:35], fmt)
                start_str = start.strftime('%Y/%m/%d')
                save_path = f'{DATA_PATH}/{start_str}/TROPOMI_{f.stem[8:]}'

                if Path(save_path).exists():
                    print('file saved already')
                    continue

                success = False
                for i in range(ATTEMPTS):
                    try:
                        dset = Dataset(f, 'r')
                    except:
                        print('failed to open nc, retrying...')
                    else:
                        success = True
                        break

                if not success:
                    print('failed to open nc, giving up.')
                    open_fails += 1
                    continue

                product = dset['/PRODUCT']

                dims = product.dimensions
                scanline = dims['scanline'].size
                ground_pixel = dims['ground_pixel'].size

                lat = product['/latitude'][0]
                lon = product['/longitude'][0]

                so2_molm2 = product['/SUPPORT_DATA/DETAILED_RESULTS/sulfurdioxide_total_vertical_column_7km']
                so2 = so2_molm2[0]
                so2 *= so2_molm2.multiplication_factor_to_convert_to_DU

                so2_flag = product['/SUPPORT_DATA/DETAILED_RESULTS/sulfurdioxide_detection_flag'][0]
                qa_value = product['/qa_value'][0]

                geo = product['/SUPPORT_DATA/GEOLOCATIONS']
                sza = geo['/solar_zenith_angle'][0]
                vza = geo['/viewing_zenith_angle'][0]

                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    save_path,
                    dims=np.array([scanline, ground_pixel]),
                    lat=lat, lon=lon, so2=so2, so2_flag=so2_flag,
                    qa_value=qa_value, sza=sza, vza=vza
                )
                print('saved.')

                dset.close()

                os.remove(f)

    return 0


if __name__ == '__main__':
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    code = download()
    exit(code)
