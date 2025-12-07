import onnx
from onnx import helper, shape_inference

def describe_onnx_model(model_path: str):
    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)

    # Basic info
    print("\n=== MODEL INFO ===")
    print(f"IR version: {model.ir_version}")
    print(f"Producer name: {model.producer_name}")
    print(f"Producer version: {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model version: {model.model_version}")
    if model.opset_import:
        for opset in model.opset_import:
            print(f"Opset domain: {opset.domain or 'ai.onnx'}  version: {opset.version}")

    # Try to infer shapes to get richer info
    try:
        inferred_model = shape_inference.infer_shapes(model)
        graph = inferred_model.graph
    except Exception as e:
        print("\n[WARN] Shape inference failed, using original graph. Reason:", e)
        graph = model.graph

    print("\n=== GRAPH INFO ===")
    print(f"Graph name: {graph.name}")
    print(f"Number of nodes: {len(graph.node)}")
    print(f"Number of initializers (parameters): {len(graph.initializer)}")

    # Inputs
    print("\n=== INPUTS ===")
    for i, inp in enumerate(graph.input):
        print(f"[Input {i}] name: {inp.name}")
        t = inp.type.tensor_type
        elem_type = t.elem_type
        dims = [d.dim_param if d.dim_param else d.dim_value for d in t.shape.dim]
        print(f"    elem_type (onnx enum): {elem_type}")
        print(f"    shape: {dims}")

    # Outputs
    print("\n=== OUTPUTS ===")
    for i, out in enumerate(graph.output):
        print(f"[Output {i}] name: {out.name}")
        t = out.type.tensor_type
        elem_type = t.elem_type
        dims = [d.dim_param if d.dim_param else d.dim_value for d in t.shape.dim]
        print(f"    elem_type (onnx enum): {elem_type}")
        print(f"    shape: {dims}")

    # List first few nodes
    print("\n=== FIRST 20 NODES (layers / ops) ===")
    for i, node in enumerate(graph.node[:20]):
        print(f"[Node {i}] op_type: {node.op_type}")
        print(f"    name: {node.name}")
        print(f"    inputs: {list(node.input)}")
        print(f"    outputs: {list(node.output)}")

    # Parameter (initializer) summary
    print("\n=== PARAMETERS (INITIALIZERS) SUMMARY ===")
    total_params = 0
    for init in graph.initializer:
        size = 1
        for dim in init.dims:
            size *= dim
        total_params += size
        print(f"name: {init.name}, dtype (onnx enum): {init.data_type}, shape: {list(init.dims)}, numel: {size}")
    print(f"\nTotal number of parameters: {total_params}")

if __name__ == "__main__":
    # Change this to the path of your model
    describe_onnx_model("tiny_conformer_conv_only_fp32_static.onnx")
