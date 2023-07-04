import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))


def check_cleargrasp():
    from lib.datasets.cleargrasp.cleargrasp_synthetic_dataset import ClearGraspSynDataset
    from lib.datasets.cleargrasp.cleargrasp_real_dataset import ClearGraspRealDataset

    data_root = './data/cleargrasp'
    import pdb
    pdb.set_trace()
    train_set = ClearGraspSynDataset(data_root=data_root,
                                     mode='train')
    data = train_set[0]
    syn_val_set = ClearGraspSynDataset(data_root=data_root,
                                       mode='val',
                                       obj_type='known')
    data = syn_val_set[0]
    syn_test_set = ClearGraspSynDataset(data_root=data_root,
                                        mode='test',
                                        obj_type='novel')
    data = syn_test_set[0]
    real_val_set = ClearGraspRealDataset(data_root=data_root,
                                         mode='val',
                                         obj_type='known')
    data = real_val_set[0]
    real_test_set = ClearGraspRealDataset(data_root=data_root,
                                          mode='test',
                                          obj_type='novel')
    data = real_test_set[0]
    print('finish...')


def check_thuman():
    from lib.datasets.thuman.thuman_dataset import THumanDataset

    data_root = './data/thuman'
    train_dataset = THumanDataset(data_root=data_root,
                                  mode='train')
    train_dataset.stat_depth()
    e = train_dataset[0]

    test_dataset = THumanDataset(data_root=data_root,
                                 mode='test')
    test_dataset.stat_depth()
    e = test_dataset[0]



if __name__ == '__main__':
    # check_cleargrasp()
    check_thuman()