---
slug: colpali-preprocessing
date: 2026-01-22T00:00:00.000Z
title: "[Notes] ColPali's Image Preprocessing Pipeline"
description: ""
tags:
  - python
  - nlp
  - pytorch
  - transformers
  - cv
keywords:
  - Python
  - NLP
  - Computer Vision
  - RAG
  - ColPali
  - Qdrant
url: /post/colpali-preprocessing/
---

## A Brief Review of Model Implementations

The ColPali class is essentially a PaliGemma model with a custom linear projection layer (the `custom_text_proj` attribute).

There are four variants of ColPali at the time of writing:

1. [vidore/colpali](https://huggingface.co/vidore/colpali)
2. [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1)
3. [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
4. [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3)

According to the [adapter_config.json](https://huggingface.co/vidore/colpali-v1.3/blob/main/adapter_config.json), they were all fine-tuned with LoRA on the same base model — [vidore/colpaligemma-3b-pt-448-base](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base).

Judging from the `target_modules` string, the fine-tuned components include all linear projection layers in the language-model portion of PaliGemma, as well as the final custom linear projection layer.

```json
{
  "target_modules": (.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)
}
```

The [vidore/colpaligemma-3b-pt-448-base](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base) model seems to be just the PaliGemma-3B model weights ported into the format used by the ColPali class.

{{< figure src="ColPali-model-class.png" caption="Analyze [the ColPali Class](https://github.com/illuin-tech/colpali/blob/51ccfe76ac2c124aa463f244390440cd313985bd/colpali_engine/models/paligemma/colpali/modeling_colpali.py#L12)" >}}

A quick glance at the newer [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0/blob/main/adapter_config.json) models shows a similar structure: a ported base model ([vidore/colqwen2.5-base](https://huggingface.co/vidore/colqwen2.5-base)) with a linear projection layer at the top, and LoRA-fine-tuned variants.

## The ColPali Image Preprocessing Pipeline

The patch size used by the ColPali model is the same as the [PaliGemma-3B model](https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json) (14 by 14 pixels), as shown in the [config.json](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base/blob/main/config.json) file of the base model. 

The vision model is a SigLIP model, whose preprocessor configurations are included in model files for all ColPali variant (e.g., [preprocessor_config.json for vidore/copali-v1.3](https://huggingface.co/vidore/colpali-v1.3/blob/main/preprocessor_config.json)). Below are the contents of the file:

```json
{
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "SiglipImageProcessor",
  "image_seq_length": 1024,
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "processor_class": "ColPaliProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 448,
    "width": 448
  }
}
```

Before we can comprehend what each of these configurations mean, we need to track down the Python class that utilizes these configurations. The preprocessor for ColPali models is [ColPaliProcessor](https://github.com/illuin-tech/colpali/blob/51ccfe76ac2c124aa463f244390440cd313985bd/colpali_engine/models/paligemma/colpali/processing_colpali.py#L10), which is based on [PaliGemmaProcessor](https://github.com/huggingface/transformers/blob/eebf8562eeeb75e693885279ad4c10a156321f28/src/transformers/models/paligemma/processing_paligemma.py#L99) from the transformers library. The PaliGemmaProcessor is a wrapper for the underlying tokenizer, chat template, and the image processor. The class used by the image processor is defined by the `image_processor_type` field in the configurations, which leads us to the [SiglipImageProcessor](https://github.com/huggingface/transformers/blob/de306e8e14672dd8392b4bd344054a6a18de8613/src/transformers/models/siglip/image_processing_siglip.py#L45) class.

### Image processing pipeline explained

The image preprocessing mostly happens in the [.preprocess](https://github.com/huggingface/transformers/blob/de306e8e14672dd8392b4bd344054a6a18de8613/src/transformers/models/siglip/image_processing_siglip.py#L108) method of `SiglipImageProcessor`. By looking at the code, we can see that the input image is firstly resized to 448 by 448, then being rescaled by 0.00392156862745098 (i.e., divided by 255 to fit in the [0, 1] domain), and finally normalized by subtracting 0.5 and dividing by 0.5.

The resulting image sequence length will be fixed (448/14=32, 32*32=1024) for all input images. This means the aspect ratio of the input image will be changed to 1:1, which is not ideal for most document images.

Let's run a simple experiment to investigate the impact of this image processor on the document images. Let's use the first page of the Attentions is All You Need paper images  as an example. Below is the uncompressed version:

{{< figure src="page-1.png" caption="Page 1; Courtesy of the[Multi-Vector Image Retrieval](https://www.deeplearning.ai/short-courses/multi-vector-image-retrieval/) course" >}}

We can run the following code to reverse the normalization and the rescaling operations in the pipeline and get the resized version of the image:

```python
from PIL import Image

from colpali_engine.models import ColPaliProcessor

model_name = "vidore/colpali-v1.3"

processor = ColPaliProcessor.from_pretrained(model_name)

images = [
  Image.open("../data/attention-is-all-you-need/page-0.png"),
]

batch_images = processor.process_images(images)

restored_pixels = ((batch_images["pixel_values"] * 0.5 + 0.5) * 255).numpy()

Image.fromarray(restored_pixels[0].transpose(1, 2, 0).astype('uint8')).save("page-0-resized.png")
```

{{< figure src="page-1-resized.png" caption="Resized Page 1" >}}

The text in the resized image is barely legible! It's impressive that the model is able to understand the text in the image. The model seems to have learned to read very blurry and slightly distorted characters during the training process.

### A closer look at the Late-Interaction mechanism

I've seemed some visualization of the interactions between the query vectors and the document vectors, such as the one from the ColiPali paper below. However, seeing the barely legible resized image from above made me want to verify the results by myself.

{{< figure src="figure-3.png" caption="Taken from Figure 3 of the ColiPali paper" >}}

Below is a visualization of the interactions of the `stacks` token vector in the query "How do the Encoder and Decoder stacks work together in Transformers?" with the image patch vectors from page 3 of the Attention is All You Need paper. I pick this token becuase it is the most clean activation map among all tokens. However, the `Transformers`, `Encoder`, and `Decoder` tokens also produce similar activation maps.

{{< figure src="ColPali-07_▁stacks_orig.jpg" caption="Activation Map on the original scale" >}}

I modified the [official plotting function](https://github.com/illuin-tech/colpali/blob/51ccfe76ac2c124aa463f244390440cd313985bd/colpali_engine/interpretability/similarity_maps.py#L13) by adding grid lines for better identifing each image patch and replacing `Image.Resampling.BICUBIC` with `Image.Resampling.NEAREST` during upscaling of the activation map to prevent the high activation (dot product) values spill into adjacent patches. This creates less aesthetically pleasing, but yet more accurate visualization.

{{< figure src="ColPali-07_▁stacks_resized.jpg" caption="Activation Map on the resized scale" >}}

The resized version shows what the model actually sees (technically speaking, it's slightly more blurry because of the lossy manipulation from matplotlib). The patches are perfect squres now. It's still baffling to me that the model somehow managed to identify the words "Stack" and "stacks" in this noisy and blurry image.

Finally, I picked the patch with the maximum activation for each query token (including the padded tokens, as specified by the paper) to visualize all the patches whose corresponding vector at the final layer contribute to the results of the late-interaction operator against this query and document pair. The result seems to make sense, most of the selected patches are around terms like "Transformer", "Stacks", "Decoder", and "Encoder."

{{< figure src="ColPali-late-interaction-activations.jpg" caption="Activation Map on the original scale" >}}

```python
new_map = np.zeros(batched_similarity_maps[0].shape[1:])
for idx in range(batched_similarity_maps[0].shape[0]):
    # Pick the `j` index that maximize the dot product result for each `i`
    index = np.unravel_index(
        batched_similarity_maps[0][idx].to(torch.float32).cpu().numpy().argmax(), 
        batched_similarity_maps[0][idx].shape
    )
    new_map[index[0]][index[1]] += batched_similarity_maps[0][idx][index[0]][index[1]]
fig, ax = plot_similarity_map(
    images[image_idx], 
    torch.tensor(new_map),
    show_grid_lines=True,
    normalization_range=(0, new_map.max())
)
ax.set_title("Late-Interaction Activations", fontsize=12)
plt.tight_layout()  # Adjust layout to fit title
fig.savefig("late-interaction-activations.jpg")`
```

## Overcoming the Restrictive Image Processing Pipeline

As we've seen above, the SigLIP image processing pipeline requires the input image to be resized to exactly 448 by 448, which poses a huge restriction on  the vision model by requiring it to infer the text content from blurry and distorted images. There are several newer variants of ColPali that aim to address this issue by switching to a vision model that allows input images in various resolutions.
