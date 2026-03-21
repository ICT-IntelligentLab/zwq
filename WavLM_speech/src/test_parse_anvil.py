from segment_prepare_iemocap import parse_anvil_file

anvil = "raw_IEMOCAP/Session1/dialog/EmoEvaluation/Ses01F_impro01_e3.anvil"
wav   = "raw_IEMOCAP/Session1/dialog/wav/Ses01F_impro01.wav"

segments = parse_anvil_file(anvil, wav)
print("Total segments:", len(segments))
print(segments[:3])