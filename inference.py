import os
from pathlib import Path
import cv2
import json
from mmdet.apis import inference_detector, init_detector
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

cls_score = 0.05


class Surgtoolloc_det():
    def __init__(self, config):
        self.config = config
        self.det_model = init_detector(config['det_cfg'], config['det_ckpt'], device='cuda:0', cfg_options={})

        self.tool_list = [
            "bipolar_dissector",
            "bipolar_forceps",
            "cadiere_forceps",
            "clip_applier",
            "force_bipolar",
            "grasping_retractor",
            "monopolar_curved_scissor",
            "needle_driver",
            "permanent_cautery_hook_spatula",
            "prograsp_forceps",
            "stapler",
            "suction_irrigator",
            "tip_up_fenestrated_grasper",
            "vessel_sealer",
        ]

    def generate_bbox(self, frame_id, frame):
        h, w, _ = frame.shape
        bboxes, labels, scores = self.gen_det_bbox(frame, self.det_model)
        if len(scores) > 0:
            inds = scores > cls_score
            if np.any(inds):
                bboxes = bboxes[inds, :4]
                labels = labels[inds]
                scores = scores[inds]
            else:
                bboxes = np.array([])
                labels = np.array([])
                scores = np.array([])
        else:
            bboxes = np.array([])
            labels = np.array([])
            scores = np.array([])

        predictions = []

        if len(labels) > 0:
            for i in range(len(labels)):
                pred_class = int(labels[i])
                score = float(scores[i])
                name = f'slice_nr_{frame_id}_' + self.tool_list[pred_class]
                pred_bbox = bboxes[i].tolist()
                bbox = [[pred_bbox[0], pred_bbox[1], 0.5],
                        [pred_bbox[2], pred_bbox[1], 0.5],
                        [pred_bbox[2], pred_bbox[3], 0.5],
                        [pred_bbox[0], pred_bbox[3], 0.5]]
                prediction = {"corners": bbox, "name": name, "probability": score}
                predictions.append(prediction)

        return predictions

    def gen_det_bbox(self, frame, det_model):
        data_sample = inference_detector(det_model, frame)
        data_sample = data_sample.cpu()
        pred_instances = data_sample.pred_instances
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        scores = pred_instances.scores
        bboxes = bboxes.numpy()
        labels = labels.numpy()
        scores = scores.numpy()
        return bboxes, labels, scores


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("endoscopic-robotic-surgery-video",): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    input_endoscopic_robotic_surgery_video = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    # resource_dir = Path("/opt/app/resources")
    # det_ckpt = os.path.join(resource_dir, "det_model.pth")
    # with open(resource_dir / "some_resource.txt", "r") as f:
    #     print(f.read())

    # # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    # model_dir = Path("/opt/ml/model")
    # with open(
    #     model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    # ) as f:
    #     print(f.read())
    # TODO: add your custom inference here

    # For now, let us make bogus predictions
    resource_dir = Path("/opt/app/resources")
    det_cfg = "/opt/app/core/det_cfg.py"
    det_ckpt = os.path.join(resource_dir, "det_model.pth")
    config = dict(det_cfg=det_cfg, det_ckpt=det_ckpt)
    processor = Surgtoolloc_det(config)
    print('Video file to be loaded: ' + str(input_endoscopic_robotic_surgery_video))
    cap = cv2.VideoCapture(str(input_endoscopic_robotic_surgery_video))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('start infering...')
    all_frames_predicted_outputs = []
    for fid in range(num_frames):
        ret, frame = cap.read()
        if ret:
            tool_detections = processor.generate_bbox(fid, frame)
            all_frames_predicted_outputs += tool_detections
    cap.release()
    # Save your output
    output_surgical_tools = dict(
        type="Multiple 2D bounding boxes", boxes=all_frames_predicted_outputs, version={"major": 1, "minor": 0}
    )
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(
        location=OUTPUT_PATH / "surgical-tools.json", content=output_surgical_tools
    )
    print('task finished.')
    print('json file generated by the submission container')

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('inputs - ', inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    print('socket slugs', socket_slugs)
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# Note to the developer:
#   the following function is very generic and should likely
#   be adopted to something more specific for your algorithm/challenge
def load_file(*, location):
    # Reads the content of a file
    with open(location) as f:
        return f.read()


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")

    available = torch.cuda.is_available()
    print(f"Torch CUDA is available: {available}")

    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")

        current_device = torch.cuda.current_device()
        print(f"\tcurrent device: {current_device}")

        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")

    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
