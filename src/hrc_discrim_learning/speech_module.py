# import opencv2 as cv\
import statistics

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
                rgb = tuple([float(v) for v in vals[0:3]])
                pdist = tuple([float(v) for v in vals[3:14]])
                color_dict[rgb] = pdist
        return color_dict

    def label_color(self, obj):
        rgb = obj.get_feature_val("rgb")
        lookup_rgb = []
        for clr in rgb:
            x = round((clr - 7.5) / 16)
            clr_approx = x * 16 + 7.5
            if clr_approx >= 255:
                clr_approx = 7.5
            lookup_rgb.append(clr_approx)

        print(lookup_rgb)

        # look up probability distribution of each color label in table
        try:
            pdist = self.color_labels[tuple(lookup_rgb)]
        except KeyError:
            print("RGB lookup error! Check speech_module.py")
            return

        pdist = list(pdist)

        # return top 2 colors and associated probabilities
        val1 = max(pdist)
        ind1 = pdist.index(val1)
        pdist.pop(ind1)
        val2 = max(pdist)
        ind2 = pdist.index(val2)

        # get color labels
        l1 = self.color_terms[ind1]
        l2 = self.color_terms[ind2]

        return (l1, ind1, val1), (l2, ind2, val2)

    def label_size(self, obj, context):
        # estimate volume based on dimensions
        dims = obj.get_feature_val("dimensions")
        target_size = 1
        for d in dims:
            target_size *= d

        # return: best label and # of stdevs from the mean in that direction
        if target_size >= context.size_xbar:
            label = "big"
        else:
            label = "small"

        diff = abs((target_size - context.size_xbar)/context.size_sd)

        return label, diff

    def label_dimensionality(self, obj, context):
        # operates on same principle as label_size - should definitely
        # be made more nuanced if possible
        x, y = obj.get_feature_val("dimensions")
        target_ratio = max(x/y, y/x)

        # return: best label and # of stdevs from the mean in that direction
        if target_ratio >= context.dim_xbar:
            label = "long"
        else:
            label = "short"

        diff = abs((target_ratio - context.dim_xbar)/context.dim_sd)

        return label, diff

if __name__ == "__main__":
    # TEST COLOR LABELLING FOR GIVEN RGB
    w2c = "w2c_4096.txt"
    sm = SpeechModule(w2c)
    # print(sm.color_labels)

    rgb = (167.500000, 7.500000, 7.500000)
    print(sm.label_color(rgb))
