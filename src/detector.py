import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class ObjectDetector:
    """
    Simple wrapper around a Hugging Face object detection model.
    """

    def __init__(
            self,
            model_name="facebook/detr-resnet-50",
            confidence_threshold=0.5
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Select available device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        # Load image processor
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        # Load model
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        )

        # Move model to selected device
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

    def detect(self, image):
        """
        Run object detection on a PIL image.

        Returns:
            List of detected objects containing:
            - label
            - score
            - box
        """

        # Convert image into model inputs
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Original image dimensions
        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.device
        )

        # Convert raw model outputs into useful results
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )

        result = results[0]

        detections = []

        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"]
        ):
            detections.append(
                {
                    "label": self.model.config.id2label[
                        label.item()
                    ],
                    "score": score.item(),
                    "box": box.tolist()
                }
            )

        return detections