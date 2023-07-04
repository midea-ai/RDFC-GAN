import os
import os.path as osp
import sys

if __name__ == '__main__':
    inputf = sys.argv[1]
    output = sys.argv[2]
    name = ['color_map', 'mask', 'albedo', 'depth_filled', 'depth_map', 'high_quality_depth']

    # absolute paths are save
    with open(output, 'a') as fout:
        path = osp.abspath(inputf)
        imagelist = os.listdir(osp.join(path, name[0]))
        for image in imagelist:

            for name in ['color_map', 'depth_map', 'high_quality_depth', 'mask']:
                fout.write(osp.join(path, name, image))
                if name != 'mask':
                    fout.write(',')
            fout.write('\n')
