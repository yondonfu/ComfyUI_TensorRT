import os

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from comfy.samplers import KSampler
from comfy.sd import load_diffusion_model_state_dict

from .models import (
    detect_version_from_model,
    get_helper_from_model,
    supported_quantization_models,
)
from .quantization import register_quant_modules, get_int8_config, filter_func, quantize_lvl, fp8_mha_disable, \
    generate_fp8_scales

MAX_RESOLUTION = 16384


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    if not os.path.exists(calib_data_path):
        raise FileNotFoundError
    with open(calib_data_path, "r", encoding="utf8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            positive_prompt=prompts[0],
            num_inference_steps=kwargs["n_steps"],
            negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
        )


class BaseQuantizer:
    @classmethod
    def INPUT_TYPES(s):
        raise NotImplementedError

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    OUTPUT_NODE = False
    CATEGORY = "TensorRT"

    def _quantize(
            self,
            model,
            clip,
            precision: str,
            calib_size: int,
            steps: int,
            calib_prompts_path="default",
            seed: int = 42,
            *args,
            **kwargs,
    ):
        model_version = detect_version_from_model(model)
        if model_version not in supported_quantization_models:
            raise NotImplementedError(f"{model_version} currently not supported.")

        if mto.ModeloptStateManager.is_converted(model.model.diffusion_model):
            print("Resetting Quantizer State")
            model = load_diffusion_model_state_dict(
                model.model.diffusion_model.state_dict()
            )
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu(
            [model], force_patch_weights=True, force_full_load=True
        )
        backbone = model.model.diffusion_model
        device = comfy.model_management.get_torch_device()

        # This is a list of prompts
        if calib_prompts_path == "default":
            calib_prompts_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "quantization/calib_prompts.txt",
            )

        calib_prompts = load_calib_prompts(1, calib_prompts_path)

        model_helper = get_helper_from_model(model)
        quant_config, extra_args = model_helper.get_qconfig(precision)
        if "quant_level" in kwargs:
            quant_level = kwargs["quant_level"]
        else:
            quant_level = extra_args.pop("quant_level", 2.5)
        if precision == "int8":
            quant_config = get_int8_config(backbone, quant_level, num_inference_steps=steps, **extra_args)
        pipe = model_helper.get_t2i_pipe(
            model, clip, seed, batch_size=1, device=device, **kwargs
        )

        def forward_loop(backbone):
            pipe.model.model.diffusion_model = backbone
            do_calibrate(
                pipe=pipe,
                calibration_prompts=calib_prompts,
                calib_size=calib_size,
                n_steps=steps,
            )

        register_quant_modules()

        mtq.quantize(backbone, quant_config, forward_loop)
        quantize_lvl(backbone, quant_level)
        mtq.disable_quantizer(backbone, filter_func)

        if precision == "fp8" and model_version not in ("Flux", "FluxSchnell"):
            generate_fp8_scales(backbone)
        if quant_level == 4.0:
            fp8_mha_disable(backbone, quantized_mha_output=False)

        # mtq.print_quant_summary(backbone)

        return (model,)


class ModelOptEzQuantizer(BaseQuantizer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "precision": (["int8", "fp8"],),
                "calib_size": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            },
            "optional": {
                "calib_prompts_path": (
                    "STRING",
                    {"forceInput": True, "default": "default"},
                ),
            },
        }

    def quantize(
            self, model, clip, precision, calib_size, steps, calib_prompts_path="default"
    ):
        return super()._quantize(
            model, clip, precision, calib_size, steps, calib_prompts_path
        )


class ModelOptQuantizer(BaseQuantizer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "precision": (["int8", "fp8"],),
                "quant_level": ([4.0, 3.0, 2.5, 2.0, 1.0], {"default": 2.5}),
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (KSampler.SAMPLERS,),
                "scheduler": (KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "percentile": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "calib_size": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "collect_method": (
                    ["min-mean", "min-max" "mean-max", "global_min", "default"],
                    {"default": "default"},
                ),
            },
            "optional": {
                "calib_prompts_path": (
                    "STRING",
                    {"forceInput": True, "default": "default"},
                ),
            },
        }

    def quantize(
            self,
            model,
            clip,
            precision,
            width,
            height,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            percentile,
            alpha,
            calib_size,
            collect_method,
            quant_level,
            calib_prompts_path="default",
    ):
        quant_level = float(quant_level)

        return super()._quantize(
            model,
            clip,
            precision,
            calib_size,
            steps,
            calib_prompts_path,
            seed=seed,
            width=width,
            height=height,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            percentile=percentile,
            alpha=alpha,
            collect_method=collect_method,
            quant_level=quant_level,
        )


NODE_CLASS_MAPPINGS = {
    "ModelOptQuantizer": ModelOptQuantizer,
    "ModelOptEzQuantizer": ModelOptEzQuantizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptQuantizer": "ModelOpt Advanced Quantizer",
    "ModelOptEzQuantizer": "ModelOpt Ez Quantizer",
}
