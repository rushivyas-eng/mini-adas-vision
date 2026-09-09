class ADASAnalyzer:
    """
    Simple ADAS analysis based on detected objects.
    """

    VEHICLES = {
        "car",
        "truck",
        "bus",
        "motorcycle"
    }

    VULNERABLE_ROAD_USERS = {
        "person",
        "bicycle"
    }

    def __init__(self, detections):
        self.detections = detections

    def count_objects(self):
        """
        Count detected objects by category.
        """

        counts = {
            "vehicles": 0,
            "people": 0,
            "bicycles": 0,
            "motorcycles": 0
        }

        for detection in self.detections:
            label = detection["label"]

            if label in {"car", "truck", "bus"}:
                counts["vehicles"] += 1

            elif label == "motorcycle":
                counts["motorcycles"] += 1

            elif label == "person":
                counts["people"] += 1

            elif label == "bicycle":
                counts["bicycles"] += 1

        return counts

    def generate_warnings(self):
        """
        Generate simple ADAS warnings.
        """

        warnings = []

        labels = {
            detection["label"]
            for detection in self.detections
        }

        if "person" in labels:
            warnings.append(
                "Pedestrian detected"
            )

        if "motorcycle" in labels:
            warnings.append(
                "Motorcycle detected"
            )

        if "bicycle" in labels:
            warnings.append(
                "Bicycle detected"
            )

        return warnings

    def analyze(self):
        """
        Generate the complete ADAS analysis.
        """

        return {
            "object_counts": self.count_objects(),
            "warnings": self.generate_warnings()
        }