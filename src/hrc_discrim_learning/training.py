#!/usr/bin/env python
import rospy
import xml.etree.ElementTree as et
import csv
from colorsys import hsv_to_rgb

from hrc_discrim_learning.base_classes import Object, Context
from hrc_discrim_learning.speech_module import SpeechModule
from hrc_discrim_learning.re_generator import REG

class CorpusTraining:
    def __init__(self):
        self.Theta = p_threshold
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        self.reg = REG()
        self.workspaces = {}

    def parse_workspace_data_from_xml(self, filename):
        self.tree = et.parse(filename)
        self.root = self.tree.getroot() # data (elements are workspaces)

        for ws in self.root:
            id = ws.attrib["id"]
            obj_lst = []
            key_item = None

            for item in ws:
                feature_dict = {}
                item_id = item.attrib["id"]

                for datum in item:
                    if datum.tag == "type":
                         feature_dict["type"] = datum.text
                    elif datum.tag == "hsv":
                        # parse as tuple
                        # print(datum.text.split(', '))
                        h, s, v = [float(x) for x in datum.text.split(', ')]
                        # normalize from gimp conventions to [0, 1] range
                        h /= 360.0
                        s /= 100.0
                        v /= 100.0
                        rgb = hsv_to_rgb(h, s, v)
                        feature_dict["rgb"] = [255 * d for d in rgb]
                    elif datum.tag == "location" or datum.tag == "dimensions":
                        x = int(datum[0].text)
                        y = int(datum[1].text)
                        feature_dict[datum.tag] = (x, y)

                o = Object()
                o.from_dict(feature_dict)
                if item_id == "KEY":
                    key_item = o

                obj_lst.append(o)

            self.workspaces[id] = (key_item, obj_lst)

    def assemble_x_for_q(self):
        color_x = []
        size_x = []
        dim_x = []
        for ws in self.workspaces:
            obj, context = self.workspaces[ws]
            clr_label, clr_score, clr_data, clr_kept = self.reg.get_model_input("color", obj, context)
            sz_label, sz_score, sz_data, sz_kept = self.reg.get_model_input("size", obj, context)
            dim_label, dim_score, dim_data, dim_kept = self.reg.get_model_input("dim", obj, context)

            color_x.append((clr_score, clr_data))
            size_x.append((sz_score, sz_data))
            dim_x.append((dim_score, dim_data))
        return color_x, size_x, dim_x

    def test_labeling(self):
        # show usage
        key, env = self.workspaces['Q3.2']
        c = Context(env)
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        clr = self.sm.label_color(key)
        sz = self.sm.label_size(key, c)
        dm = self.sm.label_dimensionality(key, c)

        print(clr, sz, dm)

    def parse_responses_from_csv(self, filename):
        with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            header = next(csvreader)

            qs_to_indicies = {}
            indicies_to_qs = {}

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                indicies_to_qs[i] = field
                qs_to_indicies[field] = i

        row = next(csvreader)
        all_responses = [[] for x in qs_to_indicies.keys()]

        for row in csvreader:
            count = 0
            for index in indicies_to_qs.keys():
                response = row[index]
                # add response to appropriate list
                all_responses[count].append(response)
                count += 1
        return all_responses

    def assemble_Y_for_q(self, all_responses_for_q):
        color_Y = []
        size_Y = []
        dim_Y = []

        for response in all_responses_for_q:
            id = self.sm.process_speech_string(response)
            color_Y.append(id[self.sm.COLOR_I])
            size_Y.append(id[self.sm.SIZE_Y])
            dim_Y.append(id[self.sm.DIM_I])

        color_avg = statistics.mean(color_Y)
        size_avg = statistics.mean(size_Y)
        dim_avg = statistics.mean(dim_Y)

        return color_avg, size_avg, dim_avg

    def process_all_outputs(self, all_responses):
        # all_responses is a list of lists;
        # each inner list is all the responses for question at that index
        color_outputs = []
        size_outputs = []
        dimension_outputs = []

        for response_set in all_responses:
            clr, sz, dim = self.assemble_Y_for_q(response_set)
            color_outputs.append((clr >= self.Theta))
            size_outputs.append((sz >= self.Theta))
            dim_outputs.append((dim >= self.Theta))

        return color_outputs, size_outputs, dim_outputs

if __name__ == "__main__":
    trainer = CorpusTraining()
    trainer.parse_workspace_data_from_xml("data/stim_v1.xml")
    trainer.test_labeling()
