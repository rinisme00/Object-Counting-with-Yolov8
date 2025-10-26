from roboflow import Roboflow
rf = Roboflow(api_key="") # add your Roboflow API key here
project = rf.workspace("test-wlzsc").project("apple-detection-jnzwb")
version = project.version(1)
dataset = version.download("yolov8")