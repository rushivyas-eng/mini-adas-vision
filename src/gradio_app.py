import io

import gradio as gr
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Rectangle
from PIL import Image

from detector import ObjectDetector
from analyzer import ADASAnalyzer

detector = ObjectDetector()

def create_annotated_image(
    image,
    spatial_results,
    danger_zone
):
    """
    Create an annotated ADAS image.

    LOW-risk detections are hidden.
    MEDIUM and HIGH-risk detections are displayed.
    Only the top 5 highest-risk detections are labeled.
    """

    width = image.width
    height = image.height

    zone_points = [
        (
            x * width,
            y * height
        )
        for x, y in danger_zone
    ]

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    ax.imshow(image)

    # Ego-lane
    polygon = Polygon(
        zone_points,
        closed=True,
        fill=False,
        linewidth=3
    )

    ax.add_patch(polygon)

    # Relevant detections
    relevant = [
        detection
        for detection in spatial_results
        if detection["risk_level"] in {
            "MEDIUM",
            "HIGH"
        }
    ]

    relevant.sort(
        key=lambda detection: detection["risk_score"],
        reverse=True
    )

    for index, detection in enumerate(relevant):

        label = detection["label"]
        confidence = detection["score"]

        xmin, ymin, xmax, ymax = detection["box"]

        center_x, center_y = detection["center"]

        bottom_x, bottom_y = detection["bottom_center"]

        risk_level = detection["risk_level"]
        risk_score = detection["risk_score"]

        proximity = detection["proximity"]

        linewidth = (
            4
            if risk_level == "HIGH"
            else 2
        )

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
            s=30
        )

        # Bottom-center point
        ax.scatter(
            bottom_x,
            bottom_y,
            s=45
        )

        # Top 5 labels only
        if index < 5:

            text = (
                f"{label} {confidence:.2f}\n"
                f"{risk_level} | {proximity.upper()}\n"
                f"risk={risk_score:.2f}"
            )

            ax.text(
                xmin,
                max(5, ymin - 5),
                text,
                fontsize=9,
                verticalalignment="bottom",
                backgroundcolor="white"
            )

    # Summary
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

    summary = (
        f"ADAS SUMMARY | "
        f"HIGH: {high_count} | "
        f"MEDIUM: {medium_count} | "
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

    ax.set_title(
        "Mini ADAS — Spatial Risk Analysis"
    )

    ax.axis("off")

    plt.tight_layout()

    # Matplotlib → PIL
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")

def analyze_image(image):
    """
    Run object detection and ADAS analysis.
    """

    if image is None:
        return (
            None,
            "### ADAS Summary\n\nPlease upload a road image.",
            "Please upload a road image."
        )

    # Object detection
    detections = detector.detect(image)

    # ADAS analysis
    analyzer = ADASAnalyzer(detections)

    analysis = analyzer.analyze(
        image.width,
        image.height
    )

    # Annotated image
    annotated_image = create_annotated_image(
        image,
        analysis["spatial_results"],
        analyzer.danger_zone
    )

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------

    high_count = sum(
        1
        for detection in analysis["spatial_results"]
        if detection["risk_level"] == "HIGH"
        and detection["inside_danger_zone"]
    )

    medium_count = sum(
        1
        for detection in analysis["spatial_results"]
        if detection["risk_level"] == "MEDIUM"
        and detection["inside_danger_zone"]
    )

    ego_lane_count = sum(
        1
        for detection in analysis["spatial_results"]
        if detection["inside_danger_zone"]
    )

    summary = (
        "### ADAS Summary\n\n"
        f"- **Detections:** {len(detections)}\n"
        f"- **High Risk:** {high_count}\n"
        f"- **Medium Risk:** {medium_count}\n"
        f"- **Ego-Lane Objects:** {ego_lane_count}"
    )

    # -----------------------------------------------
    # Warnings
    # -----------------------------------------------

    warnings = analysis["warnings"]

    if warnings:

        warning_text = "\n".join(
            f"⚠ {warning}"
            for warning in warnings
        )

    else:

        warning_text = (
            "No significant warnings."
        )

    return (
        annotated_image,
        summary,
        warning_text
    )

with gr.Blocks(
    title="Mini ADAS Vision"
) as demo:

    gr.Markdown(
        """
        # 🚗 Mini ADAS Vision

        Upload a road image to detect objects and
        perform simple spatial risk analysis.
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="Input Road Image"
        )

        output_image = gr.Image(
            type="pil",
            label="ADAS Analysis"
        )

    analyze_button = gr.Button(
        "Analyze Image"
    )

    with gr.Row():

        summary_output = gr.Markdown(
            label="ADAS Summary"
        )

        warnings_output = gr.Textbox(
            label="ADAS Warnings",
            lines=6
        )

    analyze_button.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[
            output_image,
            summary_output,
            warnings_output
        ]
    )

    clear_button = gr.ClearButton(
        components=[
            input_image,
            output_image,
            summary_output,
            warnings_output
        ],
        value="Clear"
    )


if __name__ == "__main__":
    demo.launch()