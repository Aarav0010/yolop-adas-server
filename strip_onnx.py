import onnx
import onnx.utils

new_outputs = [
    '/model.24/m.0/Conv_output_0', 
    '/model.24/m.1/Conv_output_0', 
    '/model.24/m.2/Conv_output_0',
    '2233',
    '2379'
]

onnx.utils.extract_model(
    "yolop.onnx",
    "yolop_static.onnx",
    ["images"],
    new_outputs
)
print("Successfully generated yolop_static.onnx!")
