#!/usr/bin/env python
import rospy
import xml.etree.ElementTree as et
from colorsys import hsv_to_rgb

from hrc_discrim_learning.base_classes import Object, Context
from hrc_discrim_learning.speech_module import SpeechModule

class CorpusTraining:
    def __init__(self):
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
                else:
                    obj_lst.append(o)

            self.workspaces[id] = (key_item, obj_lst)

    def test_labeling(self):
        # show usage
        key, env = self.workspaces['Q3.2']
        c = Context(env)
        w2c = "w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        clr = self.sm.label_color(key)
        sz = self.sm.label_size(key, c)
        dm = self.sm.label_dimensionality(key, c)

        print(clr, sz, dm)

if __name__ == "__main__":
    trainer = CorpusTraining()
    trainer.parse_workspace_data_from_xml("stim_v1.xml")
    trainer.test_labeling()
