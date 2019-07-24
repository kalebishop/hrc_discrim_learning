#!/usr/bin/env python
from collections import OrderedDict
class SimpleListener:
    def __init__(self):

        # absolute
        self.color              = ['red', 'green', 'blue', 'silver', 'yellow', 'black', 'pink'] # 0
        self.material           = ['plastic', 'wood', 'paper', 'ceramic', 'glass'] # 1

        self.object_type        = ['marker', 'pen', 'pencil', 'tablet', 'cup', 'bottle', 'notepad', 'computer', 'table'] # 3

        #  relative
        self.location_relative  = ['left', 'right', 'close', 'far'] # 4
        self.length_relative    = ['long', 'short'] # 5
        self.size_relative      = ['big', 'bigger', 'small', 'smaller', 'large', 'larger'] # 6

        self.term_classes = {
            "color": self.color,
            "material": self.material,
            "type": self.object_type,
            "workspace_location": self.location_relative,
            "length": self.length_relative,
            "size": self.size_relative
        }

    def term_is_relative(self, term):
        return (term in self.size_relative or term in location_relative)

    def get_term_class(self, term):
            for cls in self.term_classes:
                if term in self.term_classes[cls]:
                    return cls

    def _simple_parser(self, utterance):
        return utterance.split()

    def get_named_features(self, utterance):
        tokens = self._simple_parser(utterance)
        features = []

        for term in tokens:
            x = self.get_term_class(term)
            if x:
                features.append(x)
                # features.append(self.get_term_class(term))

        return features

    def get_named_features_as_tuples(self, utterance):
        tokens = self._simple_parser(utterance)
        features = []
        for term in tokens:
            x = self.get_term_class(term)
            if x:
                features.append((x, term))
        return features
