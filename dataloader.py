import json

import torch
from torch.utils.data import Dataset

from utils import Label2num
from utils import SepToken


class Recorddataset(Dataset):
    def __init__(self, args, path, split='train'):
        super(Recorddataset, self).__init__()
        self.args = args
        self.split = split
        self.statement = []
        self.trail1 = []
        self.trail2 = []
        self.label = []
        self.section = []
        ctpath = path + "/CT json/"
        jspath = path + f"/{split}" + ".json" if split != 'trn&dev' else path + '/train.json'
        with open(jspath) as file:
            self.data = json.load(file)
            self.uuid_list = list(self.data.keys())
        if split == 'trn&dev':  # combine tgt, then read the dev
            with open(path + '/dev.json') as file:
                self.data = {**self.data, **json.load(file)}  # combine two tgt
                self.uuid_list = list(self.data.keys())
        for id in self.uuid_list:
            self.statement.append(self.data[id]['Statement'])
            if split != 'test':
                self.label.append(Label2num[self.data[id]['Label']])
            section = self.data[id]['Section_id']
            self.section.append(section)
            with open(
                    ctpath + f"{self.data[id]['Primary_id']}" + ".json") as file:  # add the information of the first trail
                ct = json.load(file)
                trail1 = ct[section]
                self.trail1.append(self.format_change(trail1))
            if self.data[id]['Type'] == "Comparison":  # add the information of second trail
                with open(ctpath + f"{self.data[id]['Secondary_id']}" + ".json") as file:
                    ct = json.load(file)
                    trail2 = ct[section]
                    self.trail2.append(self.format_change(trail2))
            else:
                self.trail2.append("_")  # use _ to denote nothing

    def __getitem__(self, index):
        if self.args.prompt == 2:
            if self.trail2[index] == '_':
                sent = "{} [SEP] {} [SEP] {}".format(self.statement[index], self.section[index],
                                                     self.trail1[index])
            else:
                sent = "{} [SEP] {} [SEP] {} [SEP] {}".format(self.statement[index], self.section[index],
                                                              self.trail1[index], self.trail2[index])
        elif self.args.prompt == 1:
            if self.trail2[index] == '_':
                if 'gpt' in self.args.lmn:
                    sent = "{}, {}, {} <|endoftext|>".format(self.statement[index],
                                                             f"the {self.section[index]} clue of first trail is: ",
                                                             self.trail1[index])

                elif 'bart' in self.args.lmn:
                    sent = "{} {} {} {} {}".format(self.statement[index], SepToken[self.args.lmn],
                                                   f"the {self.section[index]} clue of first trail is: ",
                                                   SepToken[self.args.lmn],
                                                   self.trail1[index],
                                                   '')
                else:
                    sent = "{} {} {} {} {} {}".format(self.statement[index],
                                                      SepToken[self.args.lmn],
                                                      f"the {self.section[index]} clue of first trail is: ",
                                                      SepToken[self.args.lmn],
                                                      self.trail1[index],
                                                      SepToken[self.args.lmn])
            else:
                if 'gpt' in self.args.lmn:
                    sent = "{}, {}, {}, {}, {} <|endoftext|>".format(self.statement[index],
                                                                     f"the {self.section[index]} clue of first trail is: ",
                                                                     self.trail1[index],
                                                                     f"the {self.section[index]} clue of second trail is: ",
                                                                     self.trail2[index], )
                elif 'bart' in self.args.lmn:
                    sent = "{} {} {} {} {} {} {} {} {} {}".format(self.statement[index], SepToken[self.args.lmn],
                                                                  f"the {self.section[index]} clue of first trail is: ",
                                                                  SepToken[self.args.lmn],
                                                                  self.trail1[index],
                                                                  SepToken[self.args.lmn],
                                                                  f"the {self.section[index]} clue of second trail is: ",
                                                                  SepToken[self.args.lmn],
                                                                  self.trail2[index],
                                                                  # </s>
                                                                  '')
                else:
                    sent = "{} {} {} {} {} {} {} {} {} {}".format(self.statement[index], SepToken[self.args.lmn],
                                                                  f"the {self.section[index]} clue of first trail is: ",
                                                                  SepToken[self.args.lmn],
                                                                  self.trail1[index],
                                                                  SepToken[self.args.lmn],
                                                                  f"the {self.section[index]} clue of second trail is: ",
                                                                  SepToken[self.args.lmn],
                                                                  self.trail2[index],
                                                                  SepToken[self.args.lmn])

        elif self.args.prompt == 0:  # serve as baseline if we don't use the token
            sent = "{}, {}, {}, {}, {}".format(self.statement[index],
                                               f"the {self.section[index]} clue of first trail is: ",
                                               self.trail1[index],
                                               f"the {self.section[index]} clue of second trail is: ",
                                               self.trail2[index], )

        elif self.args.prompt == 3:
            sent = "{}".format(self.statement[index])

        else:
            raise NotImplementedError("Prompt not implemented")
        if self.split != 'test':
            return sent, torch.tensor(self.label[index])
        else:
            return sent, self.uuid_list[index]

    def __len__(self):
        return len(self.uuid_list)

    def format_change(self, sentence):
        s = ""
        for sent in sentence:
            s += sent.strip() + ","
        return s

    def get_max_length(self):
        print([len(self.__getitem__(i)[0].split(' ')) for i in range(self.__len__())])
        return max([len(self.__getitem__(i)[0].split(' ')) for i in range(self.__len__())])
