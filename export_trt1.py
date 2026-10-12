from ultralytics import YOLO

# 加载pt模型
model = YOLO("ship_best.pt")
# 导出TRT engine
model.export(format="engine", device=0, imgsz=1280, half=True)