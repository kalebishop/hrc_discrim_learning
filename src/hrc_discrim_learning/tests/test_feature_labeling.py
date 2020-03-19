from hrc_discrim_learning.speech_module import SpeechModule
from hrc_discrim_learning.base_classes import Object, Context
from hrc_discrim_learning.training import CorpusTraining

def main():
    t = CorpusTraining()
    t.parse_workspace_data_from_xml("data/stim_v2.xml")
    ws = t.workspaces['2']
    key, env = ws

    print("target object labels:")
    key_data = t.test_labeling(key, env)
    print(key_data)

    print("background object labels:")
    for obj in env.env:
        data = t.test_labeling(obj, env)
        print(data)


if __name__ == "__main__":
    main()
