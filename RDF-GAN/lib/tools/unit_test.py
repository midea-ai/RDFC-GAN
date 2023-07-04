import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
import argparse

# deprecated
# import unittest

def parse_args():
    parser = argparse.ArgumentParser('Simple test')
    parser.add_argument('--case', type=str, required=True, choices=['test_nyuv2s2d',
                                                                    'test_nyuv2r2r',
                                                                    'test_sunrgbd',
                                                                    'test_ddrnet_human'])

    args = parser.parse_args()

    return args







if __name__ == '__main__':

    args = parse_args()
    func = globals()[args.case]
    func()
