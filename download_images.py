from roboflow import Roboflow
rf = Roboflow(api_key="X7Ytt18HEuVzPuvnY1cW")
project = rf.workspace("wawan-pradana").project("cinta_v2")
dataset = project.version(1).download("yolov8")