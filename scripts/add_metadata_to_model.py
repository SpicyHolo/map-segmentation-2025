import json
import onnx

model = onnx.load('model.onnx')

class_names = {
    0: '_background',
    1: 'impervious-surfaces',
}

m1 = model.metadata_props.add()
m1.key = 'model_type'
m1.value = json.dumps('Segmentor')

m2 = model.metadata_props.add()
m2.key = 'class_names'
m2.value = json.dumps(class_names)

m3 = model.metadata_props.add()
m3.key = 'resolution'
m3.value = json.dumps(10)

# optional, if you want to standarize input after normalisation
m4 = model.metadata_props.add()
m4.key = 'standardization_mean'
m4.value = json.dumps([0.42093384, 0.43188083, 0.41308475])

m5 = model.metadata_props.add()
m5.key = 'standardization_std'
m5.value = json.dumps([0.17451693, 0.16675222, 0.163012])

onnx.save(model, 'model_out.onnx')