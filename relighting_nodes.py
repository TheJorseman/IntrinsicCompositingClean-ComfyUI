import numpy as np
import torch
from PIL import Image, ImageOps
import os
from pathlib import Path
import torchvision.transforms as T
from glob import glob

from skimage.transform import resize

from intrinsic_compositing_clean.lib.general import (
    invert, 
    uninvert, 
    view, 
    np_to_pil, 
    to2np, 
    add_chan, 
    show, 
    round_32,
    tile_imgs
)
from intrinsic_compositing_clean.lib.data_util import load_image
from intrinsic_compositing_clean.lib.normal_util import get_omni_normals

from intrinsic_compositing_clean.boosted_depth.depth_util import get_depth
#from intrinsic.model_util import load_models

from intrinsic_compositing_clean.intrinsic.pipeline import run_pipeline

from intrinsic_compositing_clean.shading.pipeline import load_reshading_model

from intrinsic_compositing_clean.shading.pipeline import (
    #load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing_clean.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

from intrinsic_compositing_clean.altered_midas.midas_net import MidasNet as IMidasNet
from intrinsic_compositing_clean.altered_midas.midas_net_custom import MidasNet_small
from intrinsic_compositing_clean.omnidata.midas.dpt_depth import DPTDepthModel
from intrinsic_compositing_clean.boosted_depth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from intrinsic_compositing_clean.boosted_depth.midas.models.midas_net import MidasNet
from argparse import Namespace

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_omni_model(omni_weigths, device="cpu"):
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(omni_weigths, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def load_models(paper_weights_path, device="cpu"):
  models = {}
  combined_dict = torch.load(paper_weights_path, map_location=device)
  stage = 1
  ord_state_dict = combined_dict['ord_state_dict']
  iid_state_dict = combined_dict['iid_state_dict']

  ord_model = IMidasNet()
  ord_model.load_state_dict(ord_state_dict)
  ord_model.eval()
  ord_model = ord_model.to(device)
  models['ord_model'] = ord_model

  iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
  iid_model.load_state_dict(iid_state_dict)
  iid_model.eval()
  iid_model = iid_model.to(device)
  models['iid_model'] = iid_model
  return models


def load_depth_pix2pix(pix2pix_path, gpu_ids=[]):
    opt = Namespace(
        Final=False,
        R0=False,
        R20=False,
        aspect_ratio=1.0,
        batch_size=1,
        checkpoints_dir=pix2pix_path,
        colorize_results=False,
        crop_size=672,
        data_dir=None,
        dataroot=None,
        dataset_mode='depthmerge',
        depthNet=None,
        direction='AtoB',
        display_winsize=256,
        epoch='latest',
        eval=True,
        generatevideo=None,
        gpu_ids=gpu_ids,
        init_gain=0.02,
        init_type='normal',
        input_nc=2,
        isTrain=False,
        load_iter=0,
        load_size=672,
        max_dataset_size=10000,
        max_res=float('inf'),
        model='pix2pix4depth',
        n_layers_D=3,
        name='mergemodel',
        ndf=64,
        netD='basic',
        netG='unet_1024',
        net_receptive_field_size=None,
        ngf=64,
        no_dropout=False,
        no_flip=False,
        norm='none',
        num_test=50,
        num_threads=4,
        output_dir=None,
        output_nc=1,
        output_resolution=None,
        phase='test',
        pix2pixsize=None,
        preprocess='resize_and_crop',
        savecrops=None,
        savewholeest=None,
        serial_batches=False,
        suffix='',
        verbose=False
    )
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()
    return pix2pixmodel


def load_depth_midas(midas_path, device="cpu"):
    midasmodel = MidasNet(midas_path, non_negative=True)
    midasmodel.to(device)
    midasmodel.eval()
    return midasmodel

def create_depth_models(midas_path, pix2pix_path, device='cpu'):
    gpu_ids = [0] if device == "cuda" else []
    pix2pixmodel = load_depth_pix2pix(pix2pix_path, gpu_ids=gpu_ids)
    midasmodel = load_depth_midas(midas_path, device=device)
    return (pix2pixmodel, midasmodel)


def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def compute_composite_normals(img, msk, model, size):
    
    bin_msk = (msk > 0)

    bb = get_bbox(bin_msk)
    bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

    # create the crop around the object in the image to send through normal net
    img_crop = img[bb[0] : bb[1], bb[2] : bb[3], :]

    crop_scale = 1024 / max(bb_h, bb_w)
    img_crop = rescale(img_crop, crop_scale)
        
    # get normals of cropped and scaled object and resize back to original bbox size
    nrm_crop = get_omni_normals(model, img_crop)
    nrm_crop = resize(nrm_crop, (bb_h, bb_w))

    h, w, c = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim
    else:
        scale = 1.0
    
    # resize to the final output size as specified by input args
    out_img = rescale(img, scale, r32=True)
    out_msk = rescale(msk, scale, r32=True)
    out_bin_msk = (out_msk > 0)
    
    # compute normals for the entire composite image at it's output size
    out_nrm_bg = get_omni_normals(model, out_img)
    
    # now the image is at a new size so the parameters of the object crop change.
    # in order to overlay the normals, we need to resize the crop to this new size
    out_bb = get_bbox(out_bin_msk)
    bb_h, bb_w = out_bb[1] - out_bb[0], out_bb[3] - out_bb[2]
    
    # now resize the normals of the crop to this size, and put them in empty image
    out_nrm_crop = resize(nrm_crop, (bb_h, bb_w))
    out_nrm_fg = np.zeros_like(out_img)
    out_nrm_fg[out_bb[0] : out_bb[1], out_bb[2] : out_bb[3], :] = out_nrm_crop

    # combine bg and fg normals with mask alphas
    out_nrm = (out_nrm_fg * out_msk[:, :, None]) + (out_nrm_bg * (1.0 - out_msk[:, :, None]))
    return out_nrm


def generate_mask_from_rgba(image):
    """
    Crea una máscara a partir de una imagen RGBA.
    
    :param image_path: Ruta a la imagen RGBA.
    :return: Imagen en escala de grises que representa la máscara.
    """
    print(image)
    # Asegurarse de que la imagen sea RGBA
    rgba_image = image.convert("RGBA")
    # Extraer el canal alfa
    alpha_channel = rgba_image.split()[3]
    # Convertir a imagen en blanco y negro para la máscara
    mask = alpha_channel.point(lambda p: 255 if p > 0 else 0)

    return mask

def apply_mask_and_combine(image1, image2, mask):
    """
    Aplica una máscara a la primera imagen y la combina con la segunda imagen.
    
    :param image1_path: Ruta a la primera imagen.
    :param image2_path: Ruta a la segunda imagen.
    :param mask: Imagen de la máscara en escala de grises.
    :param save_folder: Carpeta donde se guardará la imagen combinada.
    """
    # Asegurarse de que ambas imágenes sean del mismo tamaño
    image1 = image1.resize(image2.size)

    mask = mask.resize(image1.size)
    # Aplicar la máscara a la primera imagen
    masked_image1 = Image.composite(image1, Image.new("RGBA", image1.size), mask)

    # Combinar la imagen enmascarada con la segunda imagen
    combined_image = Image.alpha_composite(image2.convert("RGBA"), masked_image1)

    # Guardar la imagen combinada
    #base_name = os.path.basename(image2_path)
    #save_path = os.path.join(save_folder, base_name)
    combined_image = combined_image.convert("RGB")
    return combined_image

def load_image_pil(pil_img, bits=8):
    """Load an image into a numpy.array from a filepath or file object.

    params:
        path (str or file): the filepath to open or file object to load
        bits (int) optional: bit-depth of the image for normalizing to [0-1] (default 8)

    returns:
        (numpy.array): the image loaded as a numpy array
    """
    np_arr = np.array(pil_img).astype(np.single)
    return np_arr / float((2 ** bits) - 1)

class LoadImagePIL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pil_img": ("IMAGE",),
                "bits": ("INT", {"default": 8, "min": 1, "max": 16}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_ARRAY",)
    RETURN_NAMES = ("np_array",)
    FUNCTION = "load_image_pil"
    CATEGORY = "image_processing"
    
    def load_image_pil(self, pil_img, bits=8):
        """Load an image into a numpy.array from a PIL image object.
        Args:
            pil_img: PIL Image object
            bits: bit-depth for normalization
        Returns:
            numpy array normalized to [0,1]
        """
        np_arr = np.array(pil_img).astype(np.single)
        return (np_arr / float((2 ** bits) - 1),)

class ExtractSmallBgShd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("RESULT_DICT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("small_bg_shd",)
    FUNCTION = "extract_small_bg_shd"
    CATEGORY = "relighting"
    
    def extract_small_bg_shd(self, result):
        # Extrae la capa 'inv_shading' y añade una dimensión extra
        small_bg_shd = result['inv_shading'][:, :, None]
        return (small_bg_shd,)


# 1. Nodo para generar máscara a partir de una imagen RGBA
class MaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "relighting"
    
    def generate_mask(self, rgba_image):
        transform = T.ToPILImage()
        rgba_image = rgba_image.squeeze(0).permute(2, 0, 1)  # Ahora tiene forma [3, 1024, 1707]
        pil_img = transform(rgba_image)
        mask = generate_mask_from_rgba(pil_img)
        return (mask,)


def to_pil(rgb_image):
    transform = T.ToPILImage() 
    rgb_image = rgb_image.squeeze(0)
    if len(rgb_image.shape) == 3:
        rgb_image = rgb_image.permute(2, 0, 1)
    pil_img = transform(rgb_image)
    return pil_img

# 2. Nodo para combinar imágenes con máscara
class MaskApplier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "apply_mask"
    CATEGORY = "relighting"
    
    def apply_mask(self, foreground, background, mask):
        foreground = to_pil(foreground)
        background = to_pil(background)
        mask = to_pil(mask)
        mask = ImageOps.invert(mask)
        composite = apply_mask_and_combine(foreground, background, mask)
        return (composite,)

# 4. Nodo para calcular normales
class NormalsExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normals_model": ("NORMALS_MODEL",),
            }
        }
    
    RETURN_TYPES = ("NORMALS",)
    RETURN_NAMES = ("normals",)
    FUNCTION = "extract_normals"
    CATEGORY = "relighting"
    
    def extract_normals(self, image, normals_model):
        normals = get_omni_normals(normals_model, image)
        return (normals,)

# 5. Nodo para la descomposición intrínseca
class IntrinsicDecomposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intrinsic_model": ("INTRINSIC_MODEL",),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "resize_conf": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "maintain_size": ("BOOLEAN", {"default": True}),
                "linear": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("RESULT_DICT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run_intrinsic"
    CATEGORY = "relighting"
    
    def run_intrinsic(self, image, intrinsic_model, gamma, resize_conf, maintain_size, linear):
        result = run_pipeline(
            intrinsic_model,
            image ** gamma,
            resize_conf=resize_conf,
            #base_size=maintain_size,
            linear=linear,
            device=device,
        )
        return (result,)

# 6. Nodo para extraer coeficientes de luz
class LightCoeffExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shading": ("IMAGE",),
                "normals": ("NORMALS",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LIGHT_COEFFS", "LIGHT_VIS")
    RETURN_NAMES = ("coeffs", "light_visualization")
    FUNCTION = "extract_coeffs"
    CATEGORY = "relighting"
    
    def extract_coeffs(self, shading, normals, image):
        coeffs, lgt_vis = get_light_coeffs(
            shading[:, :, 0], 
            normals, 
            image
        )
        return (coeffs, lgt_vis)

# 7. Nodo para calcular normales compuestas
class CompositeNormalsCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE_ARRAY",),
                "mask_image": ("PROCESSED_MASK",),
                "normals_model": ("NORMALS_MODEL",),
                "target_size": ("INT", {"default": 1024, "min": 32, "max": 2048, "step": 32}),
            }
        }
    
    RETURN_TYPES = ("NORMALS",)
    RETURN_NAMES = ("composite_normals",)
    FUNCTION = "calculate_normals"
    CATEGORY = "relighting"
    
    def calculate_normals(self, composite_image, mask_image, normals_model, target_size):
        comp_nrm = compute_composite_normals(composite_image, mask_image, normals_model, target_size)
        return (comp_nrm,)

# 8. Nodo para calcular profundidad
class DepthEstimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_model": ("DEPTH_MODEL",),
            }
        }
    
    RETURN_TYPES = ("DEPTH",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "estimate_depth"
    CATEGORY = "relighting"
    
    def estimate_depth(self, image, depth_model):
        depth = get_depth(image, depth_model)
        return (depth,)

# 9. Nodo para armonizar albedo
class AlbedoHarmonizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "inv_shading": ("IMAGE",),
                "albedo_model": ("ALBEDO_MODEL",),
                "reproduce_paper": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("ALBEDO",)
    RETURN_NAMES = ("harmonized_albedo",)
    FUNCTION = "harmonize"
    CATEGORY = "relighting"
    
    def harmonize(self, image, mask, inv_shading, albedo_model, reproduce_paper, gamma):
        alb_harm = harmonize_albedo(
            image, 
            mask, 
            inv_shading, 
            albedo_model, 
            reproduce_paper=reproduce_paper
        ) ** gamma
        return (alb_harm,)

class ExtractInvShading:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("RESULT_DICT",),  # Espera un diccionario de resultados
            }
        }
    
    RETURN_TYPES = ("IMAGE",)  # El tipo de salida es una imagen
    RETURN_NAMES = ("inv_shading",)
    FUNCTION = "extract_inv_shading"
    CATEGORY = "relighting/extraction"
    
    def extract_inv_shading(self, result):
        """
        Extrae el componente de sombreado invertido de un diccionario de resultados.
        
        Args:
            result: Diccionario que contiene los resultados de la descomposición intrínseca
            
        Returns:
            El componente de sombreado invertido (inv_shading)
        """
        # Simplemente extrae la clave 'inv_shading' del diccionario
        inv_shd = result['inv_shading']
        
        # Se asegura de que la salida tenga el formato correcto para el siguiente procesamiento
        return (inv_shd,)




# 10. Nodo para crear imagen armonizada
class HarmonizedImageCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("ALBEDO",),
                "inv_shading": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("harmonized_image",)
    FUNCTION = "create_harmonized"
    CATEGORY = "relighting"
    
    def create_harmonized(self, albedo, inv_shading):
        harm_img = albedo * uninvert(inv_shading)[:, :, None]
        return (harm_img,)

# 11. Nodo para el proceso de reshading final
class ReshadingProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "harmonized_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "inv_shading": ("IMAGE",),
                "depth": ("DEPTH",),
                "comp_normals": ("NORMALS",),
                "albedo": ("ALBEDO",),
                "light_coeffs": ("LIGHT_COEFFS",),
                "reshading_model": ("RESHADING_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "process_reshading"
    CATEGORY = "relighting"
    
    def process_reshading(self, harmonized_image, mask, inv_shading, depth, comp_normals, 
                         albedo, light_coeffs, reshading_model):
        comp_result = compute_reshading(
            harmonized_image,
            mask,
            inv_shading,
            depth,
            comp_normals,
            albedo,
            light_coeffs,
            reshading_model
        )
        result_image = np_to_pil(comp_result['composite'])
        return ((torch.Tensor(np.array(result_image))/255).unsqueeze(0),)

class MaskPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "composite_image": ("IMAGE",),
                "bits": ("INT", {"default": 8, "min": 1, "max": 16}),
            }
        }
    
    RETURN_TYPES = ("PROCESSED_MASK",)
    RETURN_NAMES = ("mask_img",)
    FUNCTION = "preprocess_mask"
    CATEGORY = "relighting"
    
    def preprocess_mask(self, mask, composite_image, bits=8):
        # Convertir a escala de grises y redimensionar
        processed_mask = to_pil(mask).convert("L").resize(composite_image.size)
        # Convertir a array NumPy normalizado
        mask_img = np.array(processed_mask).astype(np.single) / float((2 ** bits) - 1)
        return (mask_img,)



# 12. (Opcional) Nodo de flujo completo para los que prefieren un proceso sencillo
class CompleteRelighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "car_transparent": ("IMAGE",),
                "car_mask": ("MASK",),
                "background": ("IMAGE",),
                "depth_model": ("DEPTH_MODEL",),
                "normals_model": ("NORMALS_MODEL",),
                "intrinsic_model": ("INTRINSIC_MODEL",),
                "albedo_model": ("ALBEDO_MODEL",),
                "reshading_model": ("RESHADING_MODEL",),
                "reproduce_paper": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit_image",)
    FUNCTION = "compute_relighting"
    CATEGORY = "relighting"
    
    def compute_relighting(self, car_transparent, car_mask, background, depth_model, normals_model, 
                          intrinsic_model, albedo_model, reshading_model, reproduce_paper):
        # Implementación completa del algoritmo
        car_transparent = to_pil(car_transparent)
        mask = to_pil(car_mask)
        mask = ImageOps.invert(mask)
        background = to_pil(background)
        composite = apply_mask_and_combine(car_transparent, background, mask)
        bg_img = load_image_pil(background)
        comp_img = load_image_pil(composite)
        mask_img = load_image_pil(mask.convert("L").resize(composite.size))
        
        bg_h, bg_w = bg_img.shape[:2]
        max_dim = max(bg_h, bg_w)
        scale = 512 / max_dim
        
        small_bg_img = rescale(bg_img, scale)
        small_bg_nrm = get_omni_normals(normals_model, small_bg_img)
        
        result = run_pipeline(
            intrinsic_model,
            small_bg_img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True,
            device=device,
        )
        
        small_bg_shd = result['inv_shading'][:, :, None]
        
        coeffs, lgt_vis = get_light_coeffs(
            small_bg_shd[:, :, 0], 
            small_bg_nrm, 
            small_bg_img
        )

        comp_nrm = compute_composite_normals(comp_img, mask_img, normals_model, 1024)

        h, w, c = comp_img.shape
        inference_size = 1024
        max_dim = max(h, w)
        if max_dim > inference_size:
            scale = inference_size / max_dim
        else:
            scale = 1.0
        
        img = rescale(comp_img, scale, r32=True)
        msk = rescale(mask_img, scale, r32=True)
        
        depth = get_depth(img, depth_model)
        
        result = run_pipeline(
            intrinsic_model,
            img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True,
            device=device,
        )
        inv_shd = result['inv_shading']
        
        alb_harm = harmonize_albedo(img, msk, inv_shd, albedo_model, reproduce_paper=reproduce_paper) ** 2.2
        harm_img = alb_harm * uninvert(inv_shd)[:, :, None]
        
        comp_result = compute_reshading(
            harm_img,
            msk,
            inv_shd,
            depth,
            comp_nrm,
            alb_harm,
            coeffs,
            reshading_model
        )
        result_composite = np_to_pil(comp_result['composite'])
        
        return ((torch.Tensor(np.array(result_composite))/255).unsqueeze(0),)
    
class DepthModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midas_model_path": ("STRING", {"default": "./checkpoints/depth/midas/model.pt"}),
                "pix2pix_model_path": ("STRING", {"default": "./checkpoints/depth/mergemodel/latest_net_G.pth"})
            }
        }
    
    RETURN_TYPES = ("DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/vision"
    
    def load_model(self, midas_model_path, pix2pix_model_path):
        print(f'Loading depth model from: {midas_model_path} and {pix2pix_model_path}')
        try:
            depth_model = create_depth_models(midas_model_path, pix2pix_model_path, device=device)
            print('Depth model loaded successfully!')
            return (depth_model,)
        except Exception as e:
            print(f'Error loading depth model: {str(e)}')
            raise e


class NormalsModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./checkpoints/omnidata/omnidata_dpt_normal_v2.ckpt"})
            }
        }
    
    RETURN_TYPES = ("NORMALS_MODEL",)
    RETURN_NAMES = ("normals_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/vision"
    
    def load_model(self, model_path):
        print(f'Loading normals model from: {model_path}')
        try:
            normals_model = load_omni_model(model_path, device=device)
            print('Normals model loaded successfully!')
            return (normals_model,)
        except Exception as e:
            print(f'Error loading normals model: {str(e)}')
            raise e


class IntrinsicModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./checkpoints/intrinsic/final_weights.pt"})
            }
        }
    
    RETURN_TYPES = ("INTRINSIC_MODEL",)
    RETURN_NAMES = ("intrinsic_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/vision"
    
    def load_model(self, model_path):
        print(f'Loading intrinsic decomposition model from: {model_path}')
        try:
            intrinsic_model = load_models(model_path, device=device)
            print('Intrinsic model loaded successfully!')
            return (intrinsic_model,)
        except Exception as e:
            print(f'Error loading intrinsic model: {str(e)}')
            raise e


class AlbedoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./checkpoints/albedo/albedo_paper_weights.pth"})
            }
        }
    
    RETURN_TYPES = ("ALBEDO_MODEL",)
    RETURN_NAMES = ("albedo_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/vision"
    
    def load_model(self, model_path):
        print(f'Loading albedo model from: {model_path}')
        try:
            albedo_model = load_albedo_harmonizer(model_path, device=device)
            print('Albedo model loaded successfully!')
            return (albedo_model,)
        except Exception as e:
            print(f'Error loading albedo model: {str(e)}')
            raise e


class ReshadingModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./checkpoints/shading/shading_paper_weights.pt"})
            }
        }
    
    RETURN_TYPES = ("RESHADING_MODEL",)
    RETURN_NAMES = ("reshading_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/vision"
    
    def load_model(self, model_path):
        print(f'Loading reshading model from: {model_path}')
        try:
            reshading_model = load_reshading_model(model_path, device=device)
            print('Reshading model loaded successfully!')
            return (reshading_model,)
        except Exception as e:
            print(f'Error loading reshading model: {str(e)}')
            raise e

class ImageResizerNP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE_ARRAY",),
                "target_size": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
                "round_to_32": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_image"
    CATEGORY = "relighting"
    
    def resize_image(self, image, target_size, round_to_32):
        if image.ndim == 3:
            h, w, c = image.shape
        elif image.ndim == 2:
            h, w = image.shape
            c = 1
        else:   
            raise ValueError("Unsupported image shape: {}".format(image.shape))
        max_dim = max(h, w)
        if max_dim > target_size:
            scale = target_size / max_dim
        else:
            scale = 1.0

        resized = rescale(image, scale, r32=round_to_32)
        return (resized,)


class ImageResizerNPMASK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("PROCESSED_MASK",),
                "target_size": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
                "round_to_32": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_image"
    CATEGORY = "relighting"
    
    def resize_image(self, image, target_size, round_to_32):
        if image.ndim == 3:
            h, w, c = image.shape
        elif image.ndim == 2:
            h, w = image.shape
            c = 1
        else:   
            raise ValueError("Unsupported image shape: {}".format(image.shape))
        max_dim = max(h, w)
        if max_dim > target_size:
            scale = target_size / max_dim
        else:
            scale = 1.0
        resized = rescale(image, scale, r32=round_to_32)
        return (resized,)


# 3. Nodo para redimensionar imágenes
class ImageResizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE_ARRAY",),
                "target_size": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
                "round_to_32": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_image"
    CATEGORY = "relighting"
    
    def resize_image(self, image, target_size, round_to_32):
        h, w = image.shape[:2]
        max_dim = max(h, w)
        scale = target_size / max_dim
        resized = rescale(image, scale, r32=round_to_32)
        return (resized,)

# Definiciones para ComfyUI
NODE_CLASS_MAPPINGS = {
    "DepthModelLoader": DepthModelLoader,
    "NormalsModelLoader": NormalsModelLoader,
    "IntrinsicModelLoader": IntrinsicModelLoader,
    "AlbedoModelLoader": AlbedoModelLoader,
    "ReshadingModelLoader": ReshadingModelLoader,
    "MaskGenerator": MaskGenerator,
    "MaskApplier": MaskApplier,
    "ImageResizer": ImageResizer,
    "ImageResizerNP": ImageResizerNP,
    "ImageResizerNPMASK": ImageResizerNPMASK,
    "NormalsExtractor": NormalsExtractor,
    "IntrinsicDecomposer": IntrinsicDecomposer,
    "LightCoeffExtractor": LightCoeffExtractor,
    "CompositeNormalsCalculator": CompositeNormalsCalculator,
    "DepthEstimator": DepthEstimator,
    "AlbedoHarmonizer": AlbedoHarmonizer,
    "HarmonizedImageCreator": HarmonizedImageCreator,
    "ReshadingProcessor": ReshadingProcessor,
    "CompleteRelighting": CompleteRelighting,
    "LoadImagePIL": LoadImagePIL,
    "ExtractSmallBgShd": ExtractSmallBgShd
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthModelLoader": "Depth Model Loader",
    "NormalsModelLoader": "Normals Model Loader",
    "IntrinsicModelLoader": "Intrinsic Model Loader",
    "AlbedoModelLoader": "Albedo Model Loader",
    "ReshadingModelLoader": "Reshading Model Loader",
    "MaskGenerator": "Generate Mask from RGBA",
    "MaskApplier": "Apply Mask & Combine",
    "ImageResizer": "Resize Image",
    "ImageResizerNP": "Resize Image (Numpy)",
    "ImageResizerNPMASK": "Resize Image (Numpy Mask)",
    "NormalsExtractor": "Extract Normals",
    "IntrinsicDecomposer": "Intrinsic Decomposition",
    "LightCoeffExtractor": "Extract Light Coefficients",
    "CompositeNormalsCalculator": "Calculate Composite Normals",
    "DepthEstimator": "Estimate Depth",
    "AlbedoHarmonizer": "Harmonize Albedo",
    "HarmonizedImageCreator": "Create Harmonized Image",
    "ReshadingProcessor": "Process Reshading",
    "CompleteRelighting": "Complete Relighting (All-in-One)",
    "LoadImagePIL": "Load Image PIL to Numpy Array",
    "ExtractSmallBgShd": "Extract Small Background Shading"
}


# Añadir a los mapeos existentes
NODE_CLASS_MAPPINGS["MaskPreprocessor"] = MaskPreprocessor
NODE_DISPLAY_NAME_MAPPINGS["MaskPreprocessor"] = "Preprocess Mask (Grayscale & Resize)"


NODE_CLASS_MAPPINGS["ExtractInvShading"] = ExtractInvShading
NODE_DISPLAY_NAME_MAPPINGS["ExtractInvShading"] = "Extract Inverse Shading"