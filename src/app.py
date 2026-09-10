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
    Display medium/high-risk detections together
    with the approximate ego-lane corridor.

    Low-risk detections are hidden to reduce visual noise.
    """

    width = image.width
    height = image.height

    # Convert relative danger-zone coordinates
    # into image pixel coordinates.
    zone_points = [
        (
            x * width,
            y * height
        )
        for x, y in danger_zone
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.imshow(image)

    # -------------------------------------------------
    # Draw ego-lane corridor
    # -------------------------------------------------

    polygon = Polygon(
        zone_points,
        closed=True,
        fill=False,
        linewidth=3
    )

    ax.add_patch(polygon)

    # -------------------------------------------------
    # Select relevant detections
    # -------------------------------------------------

    relevant = [
        detection
        for detection in spatial_results
        if detection["risk_level"] in {"MEDIUM", "HIGH"}
    ]

    # Sort highest risk first
    relevant.sort(
        key=lambda detection: detection["risk_score"],
        reverse=True
    )

    # -------------------------------------------------
    # Draw detections
    # -------------------------------------------------

    for index, detection in enumerate(relevant):

        label = detection["label"]
        score = detection["score"]

        xmin, ymin, xmax, ymax = detection["box"]

        center_x, center_y = detection["center"]

        risk_level = detection["risk_level"]
        risk_score = detection["risk_score"]

        vertical_zone = detection["vertical_zone"]

        # Thicker box for HIGH-risk objects
        if risk_level == "HIGH":
            linewidth = 4
        else:
            linewidth = 2

        rectangle = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=linewidth
        )

        ax.add_patch(rectangle)

        # Center point
        ax.scatter(
            center_x,
            center_y,
            s=35
        )

        # -------------------------------------------------
        # Only label the most important detections
        # -------------------------------------------------

        # Show labels for the top 8 detections.
        if index < 8:

            text = (
                f"{label} {score:.2f}\n"
                f"{vertical_zone.upper()} | {risk_level}\n"
                f"risk={risk_score:.2f}"
            )

            # Put the label slightly above the box.
            label_y = max(
                5,
                ymin - 5
            )

            ax.text(
                xmin,
                label_y,
                text,
                fontsize=8,
                verticalalignment="bottom",
                backgroundcolor="white"
            )

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