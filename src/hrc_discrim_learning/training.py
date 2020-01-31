#!/usr/bin/env python
import rospy
import xml.etree.ElementTree as et

from hrc_discrim_learning.base_classes import Object, Context

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
                        print(datum.text.split(', '))
                        feature_dict["hsv"] = tuple([int(x) for x in datum.text.split(', ')])
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


if __name__ == "__main__":
    trainer = CorpusTraining()
    trainer.parse_workspace_data_from_xml("stim_v2.xml")
