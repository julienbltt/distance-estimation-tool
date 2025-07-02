from qai_hub_models.models.midas.model import Midas
from qai_hub_models.models._shared.depth_estimation.app import DepthEstimationApp
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
from PIL import Image
import cv2
import numpy as np2


IMAGE_PATH = "captured_images/capture_20250620_155553_000.jpg"


def depth_to_distance(depth_map, min_dist=0.3, max_dist=5.0):
    # Normalise la profondeur entre min_dist et max_dist (en mètres)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return min_dist + (1.0 - depth_norm) * (max_dist - min_dist)

def show_heatmap_with_legend(heatmap, distances):
    h, w = heatmap.shape[:2]
    legend = np2.linspace(distances.min(), distances.max(), h)
    legend = np2.expand_dims(legend, axis=1)
    legend_img = cv2.applyColorMap(
        cv2.convertScaleAbs((legend - distances.min()) / (distances.max() - distances.min()) * 255, alpha=1),
        cv2.COLORMAP_INFERNO
    )
    legend_img = cv2.resize(legend_img, (40, h))
    combined = np2.hstack((heatmap, legend_img))
    cv2.putText(combined, f"{distances.max():.1f}m", (w+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(combined, f"{distances.min():.1f}m", (w+5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return combined

def get_model(type: str = "cpu", model_path: str | None = None):
    model = None

    if type == "cpu":
        # model = Midas.from_pretrained()
        model = OnnxModelTorchWrapper.OnCPU(model_path)
    elif type == "npu":
        model = OnnxModelTorchWrapper.OnNPU(model_path)
    else:
        assert Exception("Bad type enter in `get_model`function.")
    
    return model


def main_std():
    (_, _, height, width) = Midas.get_input_spec()["image"][0]
    image = Image.open(IMAGE_PATH)
    model = get_model("cpu", "midas-midas-v2-float.onnx")
    app = DepthEstimationApp(model, height, width)
    heatmap_image = app.estimate_depth(image)
    heatmap_image.show("Depth Map")


def main():
    (_, _, height, width) = Midas.get_input_spec()["image"][0]
    model = get_model("cpu", "midas-midas-v2-float.onnx")
    app = DepthEstimationApp(model, height, width)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_pil = app.estimate_depth(image)
        depth_np = np2.array(depth_pil)
        distances = depth_to_distance(depth_np)
        heatmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=255.0/depth_np.max()), cv2.COLORMAP_INFERNO)
        display_img = show_heatmap_with_legend(heatmap, distances)
        cv2.imshow("Depth Map (ESC pour quitter)", display_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()