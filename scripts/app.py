"""Local Gradio interface for research-only demonstration of predicted masks."""
import argparse
import json
from pathlib import Path

import gradio as gr
import tensorflow as tf

from brain_tumor_segmentation.inference import run_prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('models/uNet_tumor_segmentation_model.h5'))
    p.add_argument('--metadata', type=Path, help='metadata.json for new models')
    args = p.parse_args()
    if args.metadata:
        meta = json.loads(args.metadata.read_text(encoding='utf8'))
        name, size, legacy = meta['architecture'], int(meta['image_size']), False
        default_threshold = float(meta.get('threshold', 0.5))
    else:
        if args.model.name != 'uNet_tumor_segmentation_model.h5':
            p.error('--metadata is required for newly trained models')
        name, size, legacy, default_threshold = 'unet', 256, True, 0.3
    model = tf.keras.models.load_model(str(args.model), compile=False)

    def predict(rgb, threshold):
        if rgb is None:
            raise gr.Error('Upload an image first')
        return run_prediction(model, rgb, name, size, threshold, legacy)

    with gr.Blocks(title='Brain Tumor Segmentation — Research Demo') as demo:
        gr.Markdown('# Brain Tumor Segmentation — Research Demo')
        gr.Markdown('**Not for clinical use.** This is an educational prototype and does not diagnose disease. '
                    'Predicted masks may be incorrect or fail on images outside the training distribution.')
        with gr.Row():
            image = gr.Image(type='numpy', image_mode='RGB', label='Upload a brain MRI slice')
            with gr.Column():
                threshold = gr.Slider(0.05, 0.95, value=default_threshold, step=0.05, label='Segmentation threshold')
                submit = gr.Button('Generate segmentation')
        with gr.Row():
            original = gr.Image(label='Input image')
            mask = gr.Image(label='Predicted binary mask')
            overlay = gr.Image(label='Predicted mask overlay')
        submit.click(predict, inputs=[image, threshold], outputs=[original, mask, overlay])
    demo.launch(share=False)


if __name__ == '__main__':
    main()
