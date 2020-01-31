# import opencv2 as cv\
import numpy as np

class SpeechModule:
    def __init__(self, w2c_filename):
        # todo: implement svm trained on google model
        self.color_labels = self._read_color_label_w2c(w2c_filename)
        self.color_terms = ["black", "blue", "brown", "grey", "green", "orange", "pink", "purple", "red", "white", "yellow"]

    def _read_color_label_w2c(self, w2c_filename):
        color_dict = {}
        with open(w2c_filename) as rgb_file:
            for line in rgb_file:
                vals = line.split(' ')
                rgb = tuple(vals[0:3])
                pdist = tuple(vals[3:])
                color_dict[rgb] = pdist
        return color_dict

    def label_color(self, rgb):
        lookup_rgb = []
        for clr in rgb:
            x = round((clr - 7.5) / 16)
            clr_approx = x * 16 + 7.5
            if clr_approx >= 255:
                clr_approx = 7.5
            lookup_rgb.append(clr_approx)

        # look up probability distribution of each color label in table
        try:
            pdist = self.color_labels(tuple(lookup_rgb))
        except KeyError:
            print("RGB lookup error! Check speech_module.py")
            return

        # return top 2 colors and associated probabilities
        val1 = max(pdist)
        ind1 = index(val1)
        pdist.pop(ind1)
        val2 = max(pdist)
        ind2 = index(val2)

        # get color labels
        l1 = self.color_terms[ind1-3]
        l2 = self.color_terms[ind2-3]

        return (l1, val1), (l2, val2)

    def _cvt_color_to_cv_format(self, hsv_tuple):
        h = hsv_tuple[0] / 2
        s = hsv_tuple[1] / 100 * 255
        v = hsv_tuple[2] / 100 * 255
        return (h, s, v)

    def label_size(self, obj, context):
        raise NotImplementedError
        # TODO finish

        # estimate volume based on dimensions
        dims = obj.get_feature_val("dimensions")
        target_size = 1
        for d in dims:
            target_size *= d
