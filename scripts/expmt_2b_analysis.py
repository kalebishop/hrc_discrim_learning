import csv
import xml.etree.ElementTree as et
from colorsys import hsv_to_rgb
import copy
from hrc_discrim_learning.base_classes import Object, Context
from hrc_discrim_learning.speech_module import SpeechModule

FEATURES = ["color", "size", "dimensions"]

def parse_workspace_data_from_xml(filename):
    tree = et.parse(filename)
    root = tree.getroot() # data (elements are workspaces)
    all_workspaces = []

    for ws in root:
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

                if item_id == "KEY":
                    key_item = o
                    feature_dict["key"] = True
                else:
                    feature_dict["key"] = False

            o = Object()
            o.from_dict(feature_dict)

            obj_lst.append(o)

        all_workspaces.append(Context(obj_lst))
    return all_workspaces

def labeling_pass(all_workspaces, color_filename):
    sm = SpeechModule(color_filename)

    for ws in all_workspaces:
        for obj in ws.env:
            for f in FEATURES:
                # print(sm.label_feature(obj, ws, f))
                label, _ = sm.label_feature(obj, ws, f)
                obj._set_feature_val(f + "_label", label)

    return all_workspaces

def gather_feature_distribution(workspace):
    # workspace is a list of Objects
    dist_dict = {f: {} for f in FEATURES}
    dist_dict["TOTAL"] = 0
    key_obj_dict = {}

    for o in workspace.env:
        dist_dict["TOTAL"] += 1
        for f in FEATURES:
            val = o.get_feature_val(f + "_label")
            if val in dist_dict[f].keys():
                dist_dict[f][val] += 1
            else:
                dist_dict[f][val] = 1

            if o.get_feature_val("key"):
                key_obj_dict[f] = val

    return key_obj_dict, dist_dict


if __name__ == "__main__":
    all_workspaces = parse_workspace_data_from_xml("data/stim_v2.xml")
    labelled = labeling_pass(all_workspaces, "data/w2c_4096.txt")
    # print(labelled[0].env[0].features)
    key_dict, dist_dict = gather_feature_distribution(labelled[0])
    print(key_dict, dist_dict)
