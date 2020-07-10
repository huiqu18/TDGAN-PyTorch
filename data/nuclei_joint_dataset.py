
from data.base_dataset import BaseDataset, get_params, get_transform
from data.nuclei_dataset import NucleiDataset


class NucleiJointDataset(BaseDataset):

    def __init__(self, opt):
        self.split_db = []
        for i in range(opt.task_num):
            self.split_db.append(NucleiDataset(opt, i))

    def __getitem__(self, index):
        result = {}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):  # go to next dataset
                index = index - len(database)
            else:
                index_value = database[index]
                result['A'] = index_value['A']
                result['B'] = index_value['B']
                result['A_paths'] = index_value['A_paths']
                result['B_paths'] = index_value['B_paths']
                result['label_ternary'] = index_value['label_ternary']
                result['weight_map'] = index_value['weight_map']
                break
        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            length += len(i)

        return length

