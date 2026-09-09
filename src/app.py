import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from detector import ObjectDetector
from analyzer import ADASAnalyzer


def visualize_detections(image, detections):
    """
    Display the image with detected objects and bounding boxes.
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.imshow(image)

    for detection in detections:
        label = detection["label"]
        score = detection["score"]

        xmin, ymin, xmax, ymax = detection["box"]

        rectangle = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rectangle)

        ax.text(
            xmin,
            ymin,
            f"{label}: {score:.2f}",
            fontsize=10,
            backgroundcolor="white"
        )

    ax.axis("off")
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

    analysis = analyzer.analyze()

    # Print ADAS results
    print_analysis(analysis)

    # Display detections
    visualize_detections(
        image,
        detections
    )


if __name__ == "__main__":
    main()