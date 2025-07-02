from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a custom model

# Predict with the model
results = model("./data_for_test/image_22.png", save=True)  # predict on an image