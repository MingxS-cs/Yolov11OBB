from ultralytics import YOLO

# Load a model

model = YOLO("testmodel_1.pt")  # load a custom model

# Predict with the model
results = model.predict("test1.jpg",show=True,save=True)  # predict on an image


# # Access the results
# for result in results:
#     xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
#     xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
#     names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
#     confs = result.obb.conf  # confidence score of each box
#     print(f"xywhr: {xywhr}, xyxyxyxy: {xyxyxyxy}, names: {names}, confs: {confs}")