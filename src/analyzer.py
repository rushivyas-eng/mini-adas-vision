from matplotlib.path import Path


class ADASAnalyzer:
    """
    Simple ADAS analysis based on detected objects.

    The spatial and risk calculations in this class
    are heuristic approximations for learning purposes.
    They are NOT collision probabilities.
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

    OBJECT_WEIGHTS = {
        "person": 1.0,
        "motorcycle": 1.0,
        "bicycle": 1.0,
        "car": 0.7,
        "truck": 0.7,
        "bus": 0.7
    }

    VERTICAL_WEIGHTS = {
        "far": 0.3,
        "middle": 0.6,
        "near": 1.0
    }

    def __init__(
        self,
        detections,
        danger_zone=None
    ):
        self.detections = detections

        # Relative coordinates:
        # (x, y) where x and y range from 0 to 1.
        self.danger_zone = danger_zone or [
            (0.42, 0.40),
            (0.58, 0.40),
            (0.90, 1.00),
            (0.10, 1.00)
        ]

    def get_box_center(self, box):
        """
        Calculate the center of a bounding box.

        box:
            [xmin, ymin, xmax, ymax]
        """

        xmin, ymin, xmax, ymax = box

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        return center_x, center_y

    def get_box_bottom_center(self, box):
        """
        Calculate the bottom-center point of a bounding box.

        Box format:
            [xmin, ymin, xmax, ymax]

        The bottom-center represents the approximate point
        where the detected object meets the road.
        """

        xmin, ymin, xmax, ymax = box

        bottom_center_x = (xmin + xmax) / 2
        bottom_center_y = ymax

        return bottom_center_x, bottom_center_y

    def get_vertical_zone(
        self,
        center_y,
        image_height
    ):
        """
        Classify an object's vertical position.
        """

        relative_y = center_y / image_height

        if relative_y < 0.4:
            return "far"

        elif relative_y < 0.65:
            return "middle"

        else:
            return "near"

    def is_inside_danger_zone(
        self,
        center_x,
        center_y,
        image_width,
        image_height
    ):
        """
        Check whether an object's center lies
        inside the approximate ego-lane corridor.
        """

        points = [
            (
                x * image_width,
                y * image_height
            )
            for x, y in self.danger_zone
        ]

        polygon = Path(points)

        return polygon.contains_point(
            (center_x, center_y)
        )

    def calculate_proximity_score(
            self,
            bottom_y,
            box_height,
            image_height
    ):
        """
        Estimate relative proximity using the object's
        bottom-center position and bounding-box height.

        This is an image-based heuristic, NOT real-world
        distance estimation.
        """

        normalized_bottom_y = bottom_y / image_height
        normalized_height = box_height / image_height

        proximity_score = (
            0.6 * normalized_bottom_y
            + 0.4 * normalized_height
        )

        return proximity_score

    def get_proximity_level(self, proximity_score):
        """
        Convert proximity score into a simple category.
        """

        if proximity_score >= 0.50:
            return "near"

        if proximity_score >= 0.40:
            return "medium"

        return "far"

    def calculate_risk_score(
        self,
        detection,
        inside_zone,
        vertical_zone,
        proximity_score
    ):
        """
        Calculate a simple heuristic risk score.

        This is NOT a collision probability.

        Risk is composed of:
            - Object severity      : 30%
            - Ego-lane presence    : 30%
            - Proximity            : 25%
            - Detection confidence : 15%

        Final score is normalized to 0.0 - 1.0.
        """

        label = detection["label"]
        confidence = detection["score"]

        # --------------------------------------------------
        # 1. Object severity
        # --------------------------------------------------

        object_weight = self.OBJECT_WEIGHTS.get(
            label,
            0.5
        )

        # --------------------------------------------------
        # 2. Ego-lane contribution
        # --------------------------------------------------

        lane_score = 1.0 if inside_zone else 0.0

        # --------------------------------------------------
        # 3. Proximity contribution
        # --------------------------------------------------

        proximity_score = max(
            0.0,
            min(1.0, proximity_score)
        )

        # --------------------------------------------------
        # 4. Confidence contribution
        # --------------------------------------------------

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        # --------------------------------------------------
        # Weighted risk calculation
        # --------------------------------------------------

        risk_score = (
            0.30 * object_weight
            + 0.30 * lane_score
            + 0.25 * proximity_score
            + 0.15 * confidence
        )

        # --------------------------------------------------
        # Safety clamp
        # --------------------------------------------------

        risk_score = max(
            0.0,
            min(1.0, risk_score)
        )

        return risk_score


    def get_risk_level(self, score):
        """
        Convert risk score into a category.
        """

        if score >= 0.6:
            return "HIGH"

        elif score >= 0.3:
            return "MEDIUM"

        else:
            return "LOW"

    def analyze_spatial_risk(
        self,
        image_width,
        image_height
    ):
        """
        Perform spatial analysis for every detection.
        """

        spatial_results = []

        for detection in self.detections:

            center_x, center_y = self.get_box_center(
                detection["box"]
            )

            bottom_x, bottom_y = self.get_box_bottom_center(
                detection["box"]
            )

            xmin, ymin, xmax, ymax = detection["box"]

            box_width = xmax - xmin
            box_height = ymax - ymin

            proximity_score = self.calculate_proximity_score(
                bottom_y,
                box_height,
                image_height
            )

            proximity = self.get_proximity_level(
                proximity_score
            )

            vertical_zone = self.get_vertical_zone(
                center_y,
                image_height
            )

            inside_zone = self.is_inside_danger_zone(
                bottom_x,
                bottom_y,
                image_width,
                image_height
            )

            risk_score = self.calculate_risk_score(
                detection,
                inside_zone,
                vertical_zone,
                proximity_score
            )

            risk_level = self.get_risk_level(
                risk_score
            )

            spatial_results.append(
                {
                    **detection,
                    "center": (
                        center_x,
                        center_y
                    ),
                    "bottom_center": (
                        bottom_x,
                        bottom_y
                    ),
                    "vertical_zone": vertical_zone,
                    "proximity_score": proximity_score,
                    "proximity": proximity,
                    "inside_danger_zone": inside_zone,
                    "risk_score": risk_score,
                    "risk_level": risk_level
                }
            )

        return spatial_results

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

            if label in {
                "car",
                "truck",
                "bus"
            }:
                counts["vehicles"] += 1

            elif label == "motorcycle":
                counts["motorcycles"] += 1

            elif label == "person":
                counts["people"] += 1

            elif label == "bicycle":
                counts["bicycles"] += 1

        return counts

    def generate_warnings(
        self,
        spatial_results
    ):
        """
        Generate aggregated ADAS warnings.

        Multiple detections of the same object type
        are combined into a single warning.
        """

        warning_objects = {}

        for detection in spatial_results:

            label = detection["label"]
            risk_level = detection["risk_level"]
            inside_zone = detection["inside_danger_zone"]

            # Ignore objects outside the ego-lane corridor
            if not inside_zone:
                continue

            # Only MEDIUM and HIGH risk objects
            if risk_level not in {"MEDIUM", "HIGH"}:
                continue

            if label not in warning_objects:
                warning_objects[label] = {
                    "count": 0,
                    "highest_risk": risk_level
                }

            warning_objects[label]["count"] += 1

            # HIGH takes priority over MEDIUM
            if risk_level == "HIGH":
                warning_objects[label]["highest_risk"] = "HIGH"

        warnings = []

        for label, information in warning_objects.items():

            count = information["count"]
            risk_level = information["highest_risk"]

            if label == "person":
                object_name = "pedestrian"

            elif label == "motorcycle":
                object_name = "motorcycle"

            elif label == "bicycle":
                object_name = "bicycle"

            else:
                object_name = label

            if count == 1:
                warnings.append(
                    f"{risk_level}: "
                    f"{object_name.capitalize()} "
                    f"in ego-lane"
                )

            else:
                warnings.append(
                    f"{risk_level}: "
                    f"{count} {object_name}s "
                    f"in ego-lane"
                )

        return warnings

    def analyze(
        self,
        image_width=None,
        image_height=None
    ):
        """
        Generate the complete ADAS analysis.
        """

        spatial_results = None

        if (
            image_width is not None
            and image_height is not None
        ):
            spatial_results = self.analyze_spatial_risk(
                image_width,
                image_height
            )

            warnings = self.generate_warnings(
                spatial_results
            )

        else:
            warnings = []

        return {
            "object_counts": self.count_objects(),
            "spatial_results": spatial_results,
            "warnings": warnings
        }