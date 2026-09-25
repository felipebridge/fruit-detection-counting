# Model weights

Weight files (`*.pt`, `*.onnx`, ...) live here and are **not** committed to Git.

The default configuration points to `models/yolo11n.pt`. Ultralytics downloads the
official COCO-pretrained checkpoint automatically on first use.

To use a custom fruit detector, place its weights here and select them via
configuration, e.g.:

```bash
fruit-counter --input data/samples/fruit_bowl.jpg --weights models/my_fruit_detector.pt --classes ""
```

An empty class filter keeps every class the model predicts, which is what you want
for a model trained only on fruit classes.
