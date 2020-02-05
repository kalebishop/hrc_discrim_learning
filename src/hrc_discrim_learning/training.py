#!/usr/bin/env python
import rospy
import xml.etree.ElementTree as et
import csv
from colorsys import hsv_to_rgb
import statistics

from hrc_discrim_learning.base_classes import Object, Context
from hrc_discrim_learning.speech_module import SpeechModule
from hrc_discrim_learning.re_generator import REG

class CorpusTraining:
    def __init__(self):
        # self.Theta = p_threshold
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        self.reg = REG()
        self.workspaces = {}

    def train(self, xml_workspace_filename, csv_responses_filename):

        self.parse_workspace_data_from_xml(xml_workspace_filename)
        responses = self.parse_responses_from_csv(csv_responses_filename)
        tokenized_responses = self.process_all_outputs(responses)

        clr_x, sz_x, dim_x = self.assemble_x(tokenized_responses)
        clr_y, sz_y, dim_y = self.assemble_Y(tokenized_responses)

        self.reg.train_model("color", clr_x, clr_y)
        self.reg.train_model("size", sz_x, sz_y)
        self.reg.train_model("dim", dim_x, dim_y)

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

    def assemble_x_for_q(self, obj, context, tokenized_response):
        labels, tokens = tokenized_response
        type = obj.get_feature_val("type")

        color_x = []
        size_x = []
        dim_x = []

        for t in tokens:
            _, clr_score, clr_data, clr_kept_objects = self.reg.get_model_input("color", obj, context)
            _, sz_score, sz_data, sz_kept_objects = self.reg.get_model_input("size", obj, context)
            _, dim_score, dim_data, dim_kept_objects = self.reg.get_model_input("dimensions", obj, context)

            color_x.append([clr_score, clr_data])
            size_x.append([sz_score, sz_data])
            dim_x.append([dim_score, dim_data])

            if t == self.sm.COLOR_I:
                kept = clr_kept_objects
            elif t == self.sm.SIZE_I:
                kept = sz_kept_objects
            elif t == self.sm.DIM_I:
                kept = dim_kept_objects

            context = self.reg.update_context(kept)

        return color_x, size_x, dim_x

    def assemble_x(self, tokenized_responses):
        color_full_x = []
        size_full_x = []
        dim_full_x = []

        index = 0
        for ws in self.workspaces:
            obj, context = self.workspaces[ws]
            color_x, size_x, dim_x = self.assemble_x_for_q(obj, context, tokenized_responses[index])
            index+=1
            color_full_x += color_x
            size_full_x += size_x
            dim_full_x += dim_x

        return color_full_x, size_full_x, dim_full_x

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
        with open(filename) as csvfile:
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

    def assemble_Y(self, tokenized_responses):
        color_Ys = []
        size_Ys = []
        dim_Ys = []

        for labels, tokens in tokenized_responses:
            clr, sz, dim = self.assemble_Y_for_q(tokens)
            color_Ys += clr
            size_Ys += sz
            dim_Ys += dim

        return color_Ys, size_Ys, dim_Ys

    def assemble_Y_for_q(self, tokenized_response):
        color_y = []
        size_y = []
        dim_y = []

        base = [self.sm.COLOR_I, self.sm.SIZE_I, self.sm.DIM_I]
        for token in tokenized_response:
            res = map(lambda x: token == x, base)
            color_y.append(res[0])
            size_y.append(res[1])
            dim_y.append(res[2])

        if not tokenized_response:
            return [False], [False], [False]

        return color_y, size_y, dim_y

    def process_all_outputs(self, all_responses):
        chosen_responses = []
        for qid in all_responses:
            # qid is a list of all responses
            # we want the "modal" response
            selected = statistics.mode(qid)
            labels, tokens = self.sm.process_speech_string(selected)
            chosen_responses.append((labels, tokens))
        return chosen_responses

if __name__ == "__main__":
    trainer = CorpusTraining()
    # trainer.parse_workspace_data_from_xml("data/stim_v1.xml")
    # trainer.test_labeling()
    all_responses = trainer.parse_responses_from_csv("data/latest.csv")
    tokenized = trainer.process_all_outputs(all_responses)
    clr, sz, dim = trainer.assemble_Y(tokenized)
    print(len(clr) == len(sz) and len(clr) == len(dim))
    print(clr[0])
