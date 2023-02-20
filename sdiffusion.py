from __future__ import annotations

import asyncio
import functools
import importlib
import logging
from asyncio import Event, Queue
from datetime import datetime
from io import BytesIO
from logging import Logger
from pathlib import Path
from random import Random
from types import ModuleType
from typing import (Any, Dict, List, Literal, Mapping, MutableMapping,
                    Optional, Tuple, Union)

import discord
import einops
import numpy
import torch
import transformers
from bot.settings import Settings
from bot.settings.section import SettingsSection
from discord import Interaction, Member, User, Webhook
from discord.app_commands import describe
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import Size, Tensor, device
from torch.nn import Module

log: Logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


class StableDiffusion:
    """
    Component responsible for managing Stable Diffusion image generation.
    """

    #region Properties

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
        # throw error if value is not provided
        if value is None: raise Exception('No device specified for ML computation.')
        # return the config value if available, otherwise return the fallback
        return value if value is not None else fallback

    @property
    def timeout(self) -> Optional[float]:
        key: str = "timeout"
        value: Optional[str] = None
        try:
            value = self._config[key]
            return float(value) if value and isinstance(value, str) else None
        except KeyError:
            self._config[key] = ""
            return None
        except ValueError:
            self._config[key] = ""
            return None

    @timeout.setter
    def timeout(self, value: float) -> None:
        key: str = "timeout"
        if value: self._config[key] = str(value)

    #endregion


    #region Lifecycle Events

    def __init__(self, *args, **kwargs) -> None:
        """
        """
        
        self._activation: Event = Event()
        self._generation_event: Event = Event()
        self._generation_queue: Queue = Queue()
        
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
        state_dict: Mapping[str, Any] = pl_sd["state_dict"]

        yaml: Path = self.yaml if self.yaml else Path('./v1-inference.yaml')
        config = OmegaConf.load(yaml.resolve())
        if not isinstance(config, MutableMapping): raise Exception(f'{self.yaml} contains an unexpected formatting.')
        self._model: Optional[Module] = await self.__get_model__(config)
        if self._model is None: raise Exception(f'Failed to load model from {cpkt.name}')

        missing_keys, unexpected_keys = self._model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0 and verbose: log.warn(f'Missing keys: {missing_keys}')
        if len(unexpected_keys) > 0 and verbose: log.warn(f'Unexpected keys: {unexpected_keys}')

        log.info(f"CUDA {'is' if torch.cuda.is_available() else 'is not'} available.")
        log.info(f"Using {self.device_id} device.")
    
        try:
            self._model.eval()        
            self._model.to(self.device)
        except RuntimeError as error:
            log.warn(error)

        # begin loop
        await self.__start__()

    #endregion


    #region Core Loop Events

    async def __start__(self):
        """
        The core image generation loop.
        This is used internally and should not be called as a command.
        """

        while True:
            try:
                # wait for the activation event to be set
                await self._activation.wait()

                # if the model is not available
                if self._model is None:
                    # clear the activation event
                    self._activation.clear()
                    log.debug('Resetting; no model available')
                    # restart the loop
                    continue                

                log.debug('Waiting for next image generation request')
                # wait for the generation queue to return a request, or throw TimeoutError
                request: StableDiffusion.Request = await asyncio.wait_for(self._generation_queue.get(), self.timeout)

                # call dequeue hook logic
                await self.__on_dequeue__(request)

            except (asyncio.TimeoutError, Exception) as error:
                await self.__on_error__(error)


    async def __on_dequeue__(self, request: Request) -> None:
        """
        Called when a request is retrieved from the queue.
        """

        # clear the generation event
        self._generation_event.clear()
        log.debug(f"Beginning generation '{request.prompt}'")

        # generate image data given the provided prompt, preset, and steps values
        bytes: BytesIO = await self.__generate__(request.prompt, request.preset, request.steps)                

        # generate an embed given the request and image data
        data: Tuple[discord.Embed, discord.File] = self.__get_embed__(request, bytes)
        # send the embed and image data via webhook
        await request.webhook.send(embed=data[0], file=data[1])

        # wait for the generation event to be set
        await self._generation_event.wait()
        log.debug(f"Finishing generation '{request.prompt}'")


    async def __on_error__(self, error: Exception) -> None:
        """
        Called when an error occurs while handling a request.
        """

        try:
            log.error(error)
            pass
        finally:
            # clear the activation event
            self._activation.clear()

    #endregion


    #region Model Manipulation

    async def __get_model__(self, config: MutableMapping[Any, Any]) -> Optional[Module]:
        """
        Reads model configuration from provided config mapping and initializes
        the module.
        """

        # get the model metadata dictionary
        model_data: Dict[str, Any] = config.get('model', dict())

        # get the model's class name from the model metadata
        target: Optional[str] = model_data.get('target')
        # return None if target is invalid
        if not target or not isinstance(target, str): return None
        # get the module name and class name by splitting the target name
        module_name, class_name = target.rsplit('.', 1)
        # import the module by name
        module: ModuleType = importlib.import_module(module_name, package=None)
        # get the model's class object via the class name
        model: Any = getattr(module, class_name)

        # get model parameters from the model data
        params: Dict[Any, Any] = model_data.get('params', dict())
        # inject the device's ID into model configuration
        params = self.__inject_device_id__(params, self.device)

        # initialize the model using the given parameters
        return model(**params)
    
    
    def __inject_device_id__(self, params: Dict[Any, Any], device: device) -> Dict[Any, Any]:
        """
        Injects the supplied device's ID into the model configuration.

        Device ID field is located at `cond_stage_config.params.device` within
        the configuration.
        """

        # get the conditioning stage configuration
        cond_stage_config: Dict = params.get('cond_stage_config', dict())
        # get the conditioning stage configuration parameters
        cond_stage_config_params: Dict = cond_stage_config.get('params', dict())
        # set the conditioning stage configuration parameters device setting
        cond_stage_config_params['device'] = self.device_id
        # set the params section to the altered configuration parameters
        cond_stage_config['params'] = cond_stage_config_params
        # set the config section to the altered configuration
        params['cond_stage_config'] = cond_stage_config
        
        # return the altered params
        return params


    @torch.no_grad()
    def __infer__(self, prompt: str, parameters: StableDiffusion.Parameters, sampler: DDIMSampler, sampling_steps: int = 50) -> List[Image.Image]:

        with self._model.ema_scope():  # type: ignore
            start_code: Tensor = torch.randn(parameters.size, device=self.device)
            u = self._model.get_learned_conditioning(parameters.batch_size * [""])  # type: ignore
            c = self._model.get_learned_conditioning(parameters.batch_size * [prompt])  # type: ignore
            ddim_samples, ddim_intermediates = sampler.sample(
                S=sampling_steps,
                x_T=start_code,
                conditioning=c,
                unconditional_conditioning=u,
                batch_size=parameters.batch_size,
                shape=parameters.shape,
                unconditional_guidance_scale=parameters.scale,
                eta=parameters.eta,
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
        
    #endregion


    #region Business Logic

    async def __generate__(self, prompt: str, preset: Literal['low', 'medium', 'high'], steps: int = 50) -> BytesIO:
        """
        Generate an image given a prompt and an inference parameter preset.
        """
        
        try:
            seed: int = Random().randint(0, 99999)
            parameters: StableDiffusion.Parameters = StableDiffusion.Presets.medium(seed)
            if preset == 'low':
                parameters = StableDiffusion.Presets.low(seed)
            if preset == 'medium':
                parameters = StableDiffusion.Presets.medium(seed)
            if preset == 'high':
                parameters = StableDiffusion.Presets.high(seed)

            if not self._model: raise Exception('Model acquisition failed.')

            sampler: DDIMSampler = DDIMSampler(self._model)

            loop = asyncio.get_event_loop()
            func = functools.partial(self.__infer__, prompt, parameters, sampler, steps)
            images: List[Image.Image] = await loop.run_in_executor(None, func)
            image: Image.Image = images[0]
            if not image: raise Exception('Image generation failed.')

            binary: BytesIO = BytesIO()
            format: Literal['bmp', 'png'] = 'png'
            image.save(binary, bitmap_format=format)
            binary.seek(0)
            return binary

        except Exception as error:
            raise

    
    def __get_embed__(self, request: StableDiffusion.Request, binary: BytesIO) -> Tuple[discord.Embed, discord.File]:
        user: Union[discord.User, discord.Member] = request.user

        embed: discord.Embed = discord.Embed()

        embed.set_author(name=user.display_name, icon_url=user.avatar.url if user.avatar else None)

        embed.title =           "Stable Diffusion"
        embed.description =     request.prompt
        embed.timestamp =       request.timestamp
        embed.color =           discord.Colour.from_rgb(r=0, g=0, b=0)

        

        filename: str = '.'.join(['image', 'png'])
        image: discord.File = discord.File(fp=binary, filename=filename)
        embed.set_image(url=f'attachment://{filename}')

        return embed, image
    
    #endregion


    #region Application Commands

    @describe(prompt='Text describing what should be generated in the image')
    @describe(preset='The quality preset to use when generating the image')
    @describe(steps='The number of times to iterate over the diffusion process.')
    async def generate(self, interaction: Interaction, prompt: str, preset: Literal['low', 'medium', 'high'], steps: int = 50) -> None:
        """Generate an image given a prompt and an inference parameter preset"""
        
        await interaction.response.defer()

        request: StableDiffusion.Request = StableDiffusion.Request(prompt, preset, steps, interaction.user, interaction.followup, interaction.created_at)
        await self._generation_queue.put(request)

        await interaction.followup.send(f"Your prompt has been queued for generation.")

    #endregion


    #region Associated Classes

    class Parameters:

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
        """
        Defines presets for `StableDiffusion.Settings`
        """

        @staticmethod
        def low(seed: int) -> StableDiffusion.Parameters:
            return StableDiffusion.Parameters(
                dimensions=(256, 256),
                downsampling=8,
                batch_size=1,
                scale=4.0,
                channels=4,
                eta=0.75,
                seed=seed
            )

        @staticmethod
        def medium(seed: int) -> StableDiffusion.Parameters:
            return StableDiffusion.Parameters(
                dimensions=(384, 384),
                downsampling=8,
                batch_size=1,
                scale=6.0,
                channels=4,
                eta=0.8,
                seed=seed
            )

        @staticmethod
        def high(seed: int) -> StableDiffusion.Parameters:
            return StableDiffusion.Parameters(
                dimensions=(512, 512),
                downsampling=8,
                batch_size=1,
                scale=8.0,
                channels=4,
                eta=0.85,
                seed=seed
            )
        

    class Request():
        """
        A request object for the module to generate
        """
    
        def __init__(self, prompt: str, preset: Literal['low', 'medium', 'high'], steps: int, user: Union[User, Member], webhook: Webhook, timestamp: datetime):
            self._prompt: str = prompt
            self._preset: Literal['low', 'medium', 'high'] = preset
            self._steps: int = steps
            self._user: Union[User, Member] = user
            self._webhook: Webhook = webhook
            self._timestamp: datetime = timestamp
    
        @property
        def prompt(self) -> str:
            return self._prompt
    
        @property
        def preset(self) -> Literal['low', 'medium', 'high']:
            return self._preset
    
        @property
        def steps(self) -> int:
            return self._steps
        
        @property
        def user(self) -> Union[User, Member]:
            return self._user
        
        @property
        def webhook(self) -> Webhook:
            return self._webhook
        
        @property
        def timestamp(self) -> datetime:
            return self._timestamp
        
    #endregion
