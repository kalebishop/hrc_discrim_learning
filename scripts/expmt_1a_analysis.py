import xml.etree.ElementTree as et
import csv
from hrc_discrim_learning.base_classes import Object

# STIM_TO_EXCLUDE = ["6", "7", "8", "9", "10"]
STIM_TO_EXCLUDE = []
FEATURES = ["color", "size", "length"]

def obj_is_key(object):
    try:
        return object.get_feature_val("key")
    except KeyError:
        return False

def parse_workspace_from_xml(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    all_ws = []

    for ws in root:
        id = ws.attrib["id"] # question number associated with stimulus
        if id in STIM_TO_EXCLUDE:
            continue

        obj_lst = []

        for item in ws:
            item_dict = {}
            type = item.attrib["type"]
            key = ("key" in item.attrib.keys())

            item_dict["key"] = key
            for datum in item:
                item_dict[datum.tag] = datum.text

            o = Object()
            o.from_dict(item_dict)

            obj_lst.append(o)

        all_ws.append(obj_lst)

    return all_ws

def gather_feature_distribution(workspace):
    # workspace is a list of Objects
    dist_dict = {f: {} for f in FEATURES}
    dist_dict["TOTAL"] = 0
    key_obj_dict = {}

    for o in workspace:
        dist_dict["TOTAL"] += 1
        for f in FEATURES:
            val = o.get_feature_val(f)
            if val in dist_dict[f].keys():
                dist_dict[f][val] += 1
            else:
                dist_dict[f][val] = 1

            if o.get_feature_val("key"):
                key_obj_dict[f] = val

    return key_obj_dict, dist_dict

def process_distribution_data(key_obj_dict, dist_dict):
    count_variants = {}
    feature_simpson_indices = {}
    discrim_scores = {}

    for f in FEATURES:
        count_variants[f] = len(dist_dict[f].keys())

        # get discrim score data
        total_elim = 0
        key_val = key_obj_dict[f]

        sig = 0.0
        for val in dist_dict[f].keys():
            sig += (dist_dict[f][val] / dist_dict["TOTAL"])**2
            if val != key_val:
                total_elim += dist_dict[f][val]
        feature_simpson_indices[f] = sig
        discrim_scores[f] = total_elim

    return count_variants, feature_simpson_indices, discrim_scores

if __name__ == "__main__":
    output_filename = "data/v1_distribution_data.csv"
    filename = "data/stim_v1.xml"
    workspaces = parse_workspace_from_xml(filename)

    with open(output_filename, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["stim", "color_var", "size_var", "len_var", "color_simp", "size_simp", "len_simp",
            "color_discrim_net", "size_discrim_net", "len_discrim_net", "total_objs"])
        count = 3 # starts at Q3 in qualtrics survey (1 + 2 are directions)
        for ws in workspaces:
            key, dist = gather_feature_distribution(ws)
            count_variants, feature_simpson_indices, discrim_scores = process_distribution_data(key, dist)
            row = [count] + list(count_variants.values()) + list(feature_simpson_indices.values()) + list(discrim_scores.values()) + [dist["TOTAL"]]
            w.writerow(row)

            count += 1
