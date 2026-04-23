import numpy as np
from pathlib import Path


class ONNXExporter:

    @staticmethod
    def export(model, X_sample: np.ndarray, output_path: str = "models/model.onnx"):
        try:
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm
        except ImportError:
            print("[WARN] onnxmltools not installed. pip install onnxmltools")
            return

        initial_type = [("input", X_sample.dtype)]

        onnx_model = convert_lightgbm(
            model.model,
            initial_types=initial_type,
            target_opset=12,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Model exported to ONNX: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
