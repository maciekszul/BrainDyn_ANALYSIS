import sys
import json
from utilities import files
import os.path as op

# checking if the correct argument is used
# default JSON file with settings is "settings.json" 

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

# path to derivatives folder (output of fmriprep and freesurfer)
derivatives_path = parameters["derivatives_path"]
fmriprep_dir = op.join(derivatives_path, "fmriprep")
fs_dir = op.join(derivatives_path, "freesurfer")

# subject label extraction (fmriprep folder)
folder_list = files.get_folders_files(fmriprep_dir, wp=False)[0]
subjects = [i for i in folder_list if "sub" in i]
subjects.sort()
subject = subjects[index]
print("SUBJECT LABEL:", subject)

# get paths to important files:
