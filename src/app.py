import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image

from detector import ObjectDetector
from analyzer import ADASAnalyzer


def visualize_detections(
        image,
        spatial_results,
        danger_zone
):
    """
    Visualize ADAS-relevant detections.

    Low-risk detections are hidden.
    MEDIUM and HIGH-risk detections are displayed.
    Only the highest-risk detections received labels.
    """

    width = image.width
    height = image.height

    # ----------------------------------------------
    # Convert relative danger-zone coordinates
    # to image pixel coordinates.
    # ----------------------------------------------

    zone_points = [
        (
            x * width,
            y * height
        )
        for x, y in danger_zone
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.imshow(image)

    # ------------------------------------------
    # Draw ego-lane corridor
    # ------------------------------------------

    polygon = Polygon(
        zone_points,
        closed=True,
        fill=False,
        linewidth=3
    )

    ax.add_patch(polygon)

    # -------------------------------------------
    # Select MEDIUM and HIGH-risk detections
    # -------------------------------------------

    relevant = [
        detection
        for detection in spatial_results
        if detection["risk_level"] in {
            "MEDIUM",
            "HIGH"
        }
    ]

    # Highest risk first
    relevant.sort(
        key=lambda detection: detection["risk_score"],
        reverse=True
    )

    # ------------------------------------------------
    # Draw detections
    # ------------------------------------------------

    for index, detection in enumerate(relevant):

        label = detection["label"]
        confidence = detection["score"]

        xmin, ymin, xmax, ymax = detection["box"]

        center_x, center_y = detection["center"]

        bottom_x, bottom_y = detection["bottom_center"]

        risk_level = detection["risk_level"]
        risk_score = detection["risk_score"]

        proximity = detection["proximity"]

        # ------------------------------------
        # Risk-dependent visualization
        # ------------------------------------

        if risk_level == "HIGH":

            linewidth = 4
            label_size = 9

        else:

            linewidth = 2
            label_size = 8

        # ------------------------------------
        # Bouding box
        # ------------------------------------

        rectangle = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=linewidth
        )

        ax.add_patch(rectangle)

        # ---------------------------------
        # Center point
        # ---------------------------------

        ax.scatter(
            center_x,
            center_y,
            s=30
        )

        # --------------------------------
        # Bottom-center point
        # --------------------------------

        ax.scatter(
            bottom_x,
            bottom_y,
            s=45
        )

        # ---------------------------------------------
        # Label only top 5 highest-risk detections
        # ---------------------------------------------

        if index < 5:

            text = (
                f"{label} {confidence:.2f}\n"
                f"{risk_level} | "
                f"{proximity.upper()}\n"
                f"risk={risk_score:.2f}"
            )

            label_y = max(
                5,
                ymin - 5
            )

            ax.text(
                xmin,
                label_y,
                text,
                fontsize=label_size,
                verticalalignment="bottom",
                backgroundcolor="white"
            )

    # ------------------------------------------------
    # ADAS Summary
    # ------------------------------------------------

    high_count = sum(
        1
        for detection in spatial_results
        if detection["risk_level"] == "HIGH"
        and detection["inside_danger_zone"]
    )

    medium_count = sum(
        1
        for detection in spatial_results
        if detection["risk_level"] == "MEDIUM"
        and detection["inside_danger_zone"]
    )

    ego_lane_count = sum(
        1
        for detection in spatial_results
        if detection["inside_danger_zone"]
    )

    # ------------------------------------------------
    # Display summary
    # ------------------------------------------------

    summary = (
        f"ADAS SUMMARY | "
        f"HIGH: {high_count} "
        f"MEDIUM: {medium_count} "
        f"EGO-LANE OBJECTS: {ego_lane_count}"
    )

    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        backgroundcolor="white"
    )

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    ax.set_title(
        "Mini ADAS — Spatial Risk Analysis"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.show()


def print_analysis(analysis):
    """
    Print the ADAS analysis in a readable format.
    """

    print("\nADAS ANALYSIS")
    print("=" * 40)

    print("\nObject counts:")

    for category, count in analysis["object_counts"].items():
        print(f"  {category}: {count}")

    print("\nWarnings:")

    if analysis["warnings"]:
        for warning in analysis["warnings"]:
            print(f"  ⚠ {warning}")
    else:
        print("  No warnings")


def main():
    """
    Main application workflow.
    """

    # Project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Input image
    image_path = project_root / "images" / "road.jpg"

    print("Mini ADAS Vision")
    print("=" * 40)

    print(f"\nLoading image: {image_path}")

    image = Image.open(image_path)

    print(f"Image size: {image.size}")

    # Create detector
    detector = ObjectDetector(
        confidence_threshold=0.7
    )

    # Run object detection
    print("\nRunning object detection...")

    detections = detector.detect(image)

    print(f"Detections found: {len(detections)}")

    # Analyze detections
    analyzer = ADASAnalyzer(detections)

    analysis = analyzer.analyze(
        image_width=image.width,
        image_height=image.height
    )

    # Print ADAS results
    print_analysis(analysis)

    # Display detections
    visualize_detections(
        image,
        analysis["spatial_results"],
        analyzer.danger_zone
    )


if __name__ == "__main__":
    main()