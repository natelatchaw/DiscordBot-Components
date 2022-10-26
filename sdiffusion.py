from __future__ import annotations

import asyncio
import functools
import importlib
import logging
from io import BytesIO
from logging import Logger
from pathlib import Path
from random import Random
from types import ModuleType
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Tuple

import einops
import numpy
import torch
import transformers
from bot.settings import Settings
from bot.settings.section import SettingsSection
from discord import File, Interaction
from discord.app_commands import describe
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import Size, Tensor, device
from torch.nn import Module

log: Logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


class StableDiffusion:

    @property
    def cpkt(self) -> Optional[Path]:
        key: str = "cpkt"
        value: Optional[str] = None
        try:
            value = self._config[key]
            return Path(value).resolve() if value and isinstance(value, str) else None
        except KeyError:
            self._config[key] = ""
            return None
        except ValueError:
            self._config[key] = ""
            return None

    @property
    def yaml(self) -> Optional[Path]:
        key: str = "yaml"
        value: Optional[str] = None
        try:
            value = self._config[key]
            return Path(value).resolve() if value and isinstance(value, str) else None
        except KeyError:
            self._config[key] = ""
            return None
        except ValueError:
            self._config[key] = ""
            return None

    @property
    def device_id(self) -> str:
        """
        Reads the `device_id` flag from configuration
        """
        # define a fallback with the most optimal supported device
        fallback: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        # define the configuration key
        key: str = "device_id"
        # get the string stored in the config, if available
        value: Optional[str] = self._config.get_string(key)
        # return the config value if available, otherwise return the fallback
        return value if value is not None else fallback

    def __init__(self, *args, **kwargs) -> None:
        """
        """
        
        try:
            self._settings: Settings = kwargs['settings']
        except KeyError as error:
            raise Exception(f'Key {error} was not found in provided kwargs')

        # create a config section for this component
        self._settings.client[self.__class__.__name__] = SettingsSection(self.__class__.__name__, self._settings.client._parser, self._settings.client._reference)
        # create reference to this component's config section
        self._config: SettingsSection = self._settings.client[self.__class__.__name__]  # type: ignore

        #
        self.device: device = torch.device(self.device_id)


    async def __setup__(self):
        verbose: bool = False

        cpkt: Path = self.cpkt if self.cpkt else Path('./sd-v1-4.ckpt')
        pl_sd: Any = torch.load(cpkt, map_location='cpu')
        global_step: str = pl_sd['global_step']        
        state_dict = pl_sd["state_dict"]

        yaml: Path = self.yaml if self.yaml else Path('./v1-inference.yaml')
        config = OmegaConf.load(yaml.resolve())
        self._model: Optional[Module] = await self.__get_model__(config)  # type: ignore
        if self._model is None: raise Exception(f'Failed to load model from {cpkt.name}')

        m, u = self._model.load_state_dict(state_dict, strict=False)
        if len(m) > 0 and verbose: print(f'missing keys: {m}')
        if len(u) > 0 and verbose: print(f'unexpected keys: {u}')

        log.info(f"CUDA {'is' if torch.cuda.is_available() else 'is not'} available.")
        log.info(f"Using {self.device_id} device.")
    
        try:
            self._model.eval()        
            self._model.to(self.device)
        except RuntimeError as error:
            log.warn(error)


    @describe(prompt='Text describing what should be generated in the image')
    @describe(preset='The quality preset to use when generating the image')
    @describe(steps='The number of times to iterate over the diffusion process.')
    async def generate(self, interaction: Interaction, prompt: str, preset: Literal['low', 'medium', 'high'], steps: int = 50) -> None:
        """Generate an image given a prompt and an inference parameter preset"""
        
        await interaction.response.defer()

        try:
            seed: int = Random().randint(0, 99999)
            settings: StableDiffusion.Settings = StableDiffusion.Presets.medium(seed)
            if preset == 'low':
                settings = StableDiffusion.Presets.low(seed)
            if preset == 'medium':
                settings = StableDiffusion.Presets.medium(seed)
            if preset == 'high':
                settings = StableDiffusion.Presets.high(seed)

            seed_everything(settings.seed)

            if not self._model: raise Exception('Model acquisition failed.')

            sampler: DDIMSampler = DDIMSampler(self._model)

            loop = asyncio.get_event_loop()
            func = functools.partial(self.__infer__, prompt, settings, sampler, steps)
            images: List[Image.Image] = await loop.run_in_executor(None, func)
            image: Image.Image = images[0]
            if not image: raise Exception('Image generation failed.')

            output_binary: BytesIO = BytesIO()
            extension: str = 'PNG'
            image.save(output_binary, extension)
            output_binary.seek(0)
            file: File = File(fp=output_binary, filename=f'output.{extension}')
            await interaction.followup.send(file=file)

        except Exception as error:
            await interaction.followup.send(f'{error}')
            raise


    @torch.no_grad()
    #@torch.autocast(DEVICE_TYPE)
    def __infer__(self, prompt: str, settings: StableDiffusion.Settings, sampler: DDIMSampler, sampling_steps: int = 50) -> List[Image.Image]:

        with self._model.ema_scope():  # type: ignore
            start_code: Tensor = torch.randn(settings.size, device=self.device)
            u = self._model.get_learned_conditioning(settings.batch_size * [""])  # type: ignore
            c = self._model.get_learned_conditioning(settings.batch_size * [prompt])  # type: ignore
            ddim_samples, ddim_intermediates = sampler.sample(
                S=sampling_steps,
                x_T=start_code,
                conditioning=c,
                unconditional_conditioning=u,
                batch_size=settings.batch_size,
                shape=settings.shape,
                unconditional_guidance_scale=settings.scale,
                eta=settings.eta,
                verbose=False,
            )

            x_ddim_samples: Tensor = self._model.decode_first_stage(ddim_samples) # type: ignore
            x_ddim_samples = torch.clamp((x_ddim_samples + 1.0) / 2.0, min=0.0, max=1.0)

            outputs: List[Image.Image] = list()

            for x_ddim_sample in x_ddim_samples:
                sample: Tensor = 255.0 * einops.rearrange(x_ddim_sample.cpu().numpy(), 'c h w -> h w c')
                output: Image.Image = Image.fromarray(sample.astype(numpy.uint8)) # type: ignore
                outputs.append(output)

            return outputs


    async def __get_model__(self, config: MutableMapping[Any, Any]) -> Optional[Module]:
        
        # get the model metadata dictionary
        model_data: Dict[str, Any] = config.get('model', dict())
        # get the model's class name from the model metadata
        target: Optional[str] = model_data.get('target')
        # return None if no target is defined
        if target is None: return None
        # get the module name and class name by splitting the target name
        module_name, class_name = target.rsplit('.', 1)
        # import the module by name
        module: ModuleType = importlib.import_module(module_name, package=None)
        # get the model's class object via the class name
        model: Any = getattr(module, class_name)
        # get model parameters from the model data
        params: Dict = model_data.get('params', dict())

        # get the conditioning stage config section
        params['cond_stage_config']: Dict = params.get('cond_stage_config', dict())
        # get the conditioning stage parameters config section
        params['cond_stage_config']['params']: Dict = params['cond_stage_config'].get('params', dict())
        # set the device id
        params['cond_stage_config']['params']['device']: str = self.device_id

        # instantiate the model using the given parameters
        return model(**params)



    class Settings:

        def __init__(self, *, dimensions: Tuple[int, int], downsampling: int, batch_size: int, scale: float, channels: int, eta: float, seed: int):
            self._height: int = dimensions[0]
            self._width: int = dimensions[1]
            self._downsampling: int = downsampling
            self._batch_size: int = batch_size
            self._scale: float = scale
            self._channels: int = channels
            self._eta: float = eta
            self._seed: int = seed

        @property
        def height(self) -> int: return self._height

        @property
        def width(self) -> int: return self._width

        @property
        def downsampling(self) -> int: return self._downsampling

        @property
        def batch_size(self) -> int: return self._batch_size

        @property
        def scale(self) -> float: return self._scale

        @property
        def channels(self) -> int: return self._channels

        @property
        def eta(self) -> float: return self._eta

        @property
        def seed(self) -> int: return self._seed

        @property
        def shape(self) -> List[int]: return [
            self._channels,
            self._height // self._downsampling,
            self._width // self._downsampling
        ]

        @property
        def size(self) -> Size: return Size([
            self._batch_size,
            self._channels,
            self._height // self._downsampling,
            self._width // self._downsampling
        ])


    class Presets:

        @staticmethod
        def low(seed: int) -> StableDiffusion.Settings:
            return StableDiffusion.Settings(
                dimensions=(256, 256),
                downsampling=8,
                batch_size=1,
                scale=4.0,
                channels=4,
                eta=0.75,
                seed=seed
            )

        @staticmethod
        def medium(seed: int) -> StableDiffusion.Settings:
            return StableDiffusion.Settings(
                dimensions=(384, 384),
                downsampling=8,
                batch_size=1,
                scale=6.0,
                channels=4,
                eta=0.8,
                seed=seed
            )

        @staticmethod
        def high(seed: int) -> StableDiffusion.Settings:
            return StableDiffusion.Settings(
                dimensions=(512, 512),
                downsampling=8,
                batch_size=1,
                scale=8.0,
                channels=4,
                eta=0.85,
                seed=seed
            )

