import asyncio
import logging
from asyncio import Event, Queue
from logging import Logger
from typing import Any, Dict, List, Optional, Union

import discord
import youtube_dl
from discord import Interaction
from discord.app_commands import Range, describe
from configuration.section import Section
from settings.settings import Settings

from components.models.audio import (AudioLogger, InvalidChannelError,
                                     Metadata, Request)

log: Logger = logging.getLogger(__name__)


AUDIO_LOGGER: AudioLogger = AudioLogger()

DEFAULTS: Dict[str, Any] = {
    'format': 'bestaudio/best',
    'noplaylist': False,
    'postprocessors': [
        {
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'opus',
        },
    ],
    'logger': AUDIO_LOGGER,
    'progress_hooks': [ ],
}


class Audio():
    """
    Component responsible for audio playback.
    """

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


    def __init__(self, *args, **kwargs):
        """
        Initializes state management objects for the audio loop
        """

        self._connection: Event = Event()
        self._playback_event: Event = Event()
        self._playback_queue: Queue = Queue()
        self._client: Optional[discord.VoiceClient] = None
        self._current: Optional[Metadata] = None
        
        try:
            self._settings: Settings = kwargs['settings']
        except KeyError as error:
            raise Exception(f'Key {error} was not found in provided kwargs')


    async def __setup__(self) -> None:
        """
        Called after instance properties are initialized.
        """

        # create a config section for Audio
        self._settings.client[self.__class__.__name__] = Section(self.__class__.__name__, self._settings.client._parser, self._settings.client._reference)
        # create reference to Audio config section
        self._config: Section = self._settings.client[self.__class__.__name__]
        # begin audio loop
        await self.__start__()


    def __on_complete__(self, error: Optional[Exception]):
        """
        Called when a request completes in the audio loop.
        """

        # set the playback event
        self._playback_event.set()
        # log error if it was provided
        if error: log.error(error)


    def __on_dequeue__(self, request: Request) -> None:
        """
        Called when a request is retrieved from the queue.
        """

        # set the current metadata
        self._current = request.metadata


    async def __start__(self):
        """
        The core audio playback loop.
        This is used internally and should not be called as a command.
        """

        while True:
            try:
                # wait for the connection event to be set
                await self._connection.wait()
                log.debug(f"Beginning core audio playback loop")

                # if the voice client is not available
                if self._client is None:
                    # clear the connection event
                    self._connection.clear()
                    log.debug(f"Resetting; no voice client available")
                    # restart the loop
                    continue

                log.debug(f"Waiting for next audio request")
                # wait for the playback queue to return a request, or throw TimeoutError
                request: Request = await asyncio.wait_for(self._playback_queue.get(), self.timeout)

                # call dequeue hook logic
                self.__on_dequeue__(request)

                # clear the playback event
                self._playback_event.clear()
                log.debug(f"Beginning track '{request.metadata.title}'")

                # play the request
                self._client.play(request.source, after=self.__on_complete__)

                # wait for the playback event to be set
                await self._playback_event.wait()
                log.debug(f"Finishing track '{request.metadata.title}'")

            except asyncio.TimeoutError as error:
                log.error(error)
                try:
                    await self.__disconnect__()
                finally:
                    # clear the connection event
                    self._connection.clear()

            except Exception as error:
                log.error(error)
                try:
                    await self.__disconnect__()
                finally:
                    # clear the connection event
                    self._connection.clear()

    
    async def __connect__(self, interaction: Interaction):
        """
        Connects the bot to the user's voice channel.
        
        ### Raises
        - discord.ClientException:
            You are already connected to a voice channel.
        """

        try:
            # get the user's voice state
            state: Optional[discord.VoiceState] = interaction.user.voice  # type: ignore
            # if the user doesn't have a voice state, raise error
            if not state:
                raise InvalidChannelError(None)

            # if the voice state does not reference a channel, raise error
            if not state.channel:
                raise InvalidChannelError(state.channel)

            # connect to the channel and get a voice client
            self._client = await state.channel.connect()
            # set the connection event
            self._connection.set()

        except discord.ClientException as exception:
            log.warn(exception)
            # ignore client exception errors
            pass

        finally:
            pass


    async def __disconnect__(self, *, force: bool = False):
        """
        Disconnects the bot from the joined voice channel.
        """

        try:
            # if the voice client is unavailable
            if not self._client:
                raise ConnectionError("Voice connection unavailable")
            # if the voice client is not connected
            if not self._client.is_connected():
                raise ConnectionError("Voice connection disconnected")

            # disconnect the voice client and clear the voice client
            self._client = await self._client.disconnect(force=force)
            # clear the connection event
            self._connection.clear()
            
        except Exception as exception:
            log.warn(exception)
            pass
        
        finally:
            pass

    
    async def __query__(self, interaction: Interaction, query: str, *, downloader: Optional[youtube_dl.YoutubeDL] = None) -> Optional[Metadata]:
        """
        Searches for a query on YouTube and downloads the metadata.

        Parameters:
            - query: A string or URL to download metadata from YouTube
            - downloader: YoutubeDL downloader instance
        """
        
        # use provided downloader or initialize one if not provided
        downloader = downloader if downloader else youtube_dl.YoutubeDL(DEFAULTS)
        # extract info for the provided query
        data: Optional[Dict[str, Any]] = downloader.extract_info(query, download=False)

        # get the entries property, if it exists
        entries: Optional[List[Any]] = data.get('entries') if data else None
        # if the data contains a list of entries, use the list;
        # otherwise create list from data (single entry)
        results: List[Optional[Dict[str, Any]]] = entries if entries else [data]
        # return the first available result
        result: Optional[Dict[str, Any]] = results[0]

        # return a Metadata object if result exists
        return Metadata.__from_dict__(interaction, result) if result else None


    async def __queue__(self, interaction: Interaction, metadata: Metadata, options: List[str] = list()) -> Request:
        """
        Adds metadata to the queue.
        """

        # add the audio filter parameter to the options list
        options.append(r'-vn')

        # create source from metadata url and options
        source: discord.AudioSource = discord.FFmpegOpusAudio(metadata.source, options=' '.join(options))
        # create request from source and metadata
        request: Request = Request(source, metadata)

        # add the request to the queue
        await self._playback_queue.put(request)
        # return the request
        return request


    async def __pause__(self, interaction: Interaction) -> None:
        """
        Pauses the current track.
        """
        # pause the voice client if available
        if self._client: self._client.pause()
        #
        return


    async def __skip__(self, interaction: Interaction) -> Optional[Metadata]:
        """
        Skips the current track.
        """

        # store reference to the current metadata
        skipped: Optional[Metadata] = self._current
        # stop the voice client if available
        if self._client: self._client.stop()
        # return the skipped metadata
        return skipped

    
    async def __stop__(self, interaction: Interaction) -> None:
        """
        Stops audio playback.
        """

        # stop the voice client if available
        if self._client: self._client.stop()
        # disconnect the voice client
        await self.__disconnect__()
        #
        return

    @describe(query='The audio track to search for')
    @describe(speed='A numerical modifier to alter the audio speed by')
    async def play(self, interaction: Interaction, query: str, speed: Range[float, 0.5, 2.0] = 1.0) -> None:
        """
        Plays audio in a voice channel
        """

        followup: discord.Webhook = interaction.followup
        await interaction.response.defer(ephemeral=False, thinking=True)

        options: List[str] = list()

        try:
            await self.__connect__(interaction)
        except discord.ClientException as exception:
            log.warn(exception)
        except Exception as exception:
            await followup.send(f'{exception}')
            return

        try:
            # download metadata for the provided query
            metadata: Optional[Metadata] = await self.__query__(interaction, f'ytsearch:{query}')
            if not metadata: raise Exception(f"No result found for '{query}'.")

            # get multiplier if speed was provided
            multiplier: Optional[float] = float(speed) if speed else None
            # assert the multiplier is within supported bounds
            multiplier = multiplier if multiplier and multiplier > 0.5 and multiplier < 2.0 else None
            # if a multiplier was specified, add it to the options list
            if multiplier and multiplier != 1.0: options.append(rf'-filter:a "atempo={multiplier}"')

            # queue the song and get the song's request data
            request: Optional[Request] = await self.__queue__(interaction, metadata, options=options)

            # generate an embed from the song request data
            embed: discord.Embed = self.__get_embed__(interaction, request.metadata)

            # send the embed
            await followup.send(embed=embed)

        except Exception as exception:
            await followup.send(f'{exception}')
            raise

    async def skip(self, interaction: discord.Interaction) -> None:
        """
        Skips the currently playing song
        """

        followup: discord.Webhook = interaction.followup
        await interaction.response.defer(ephemeral=False, thinking=True)

        # skip the song and get the skipped song's metadata.
        metadata: Optional[Metadata] = await self.__skip__(interaction)

        # generate an embed from the song request data
        embed: Optional[discord.Embed] = self.__get_embed__(interaction, metadata) if metadata else None
        # send the embed
        await followup.send(f'Skipped {metadata.title if metadata else "Missing Title"}')

    
    def __get_embed__(self, interaction: discord.Interaction, metadata: Metadata) -> discord.Embed:
        user: Union[discord.User, discord.Member] = interaction.user

        embed: discord.Embed = discord.Embed()

        embed.set_author(name=user.display_name, icon_url=user.avatar.url if user.avatar else None)
        embed.set_image(url=metadata.thumbnail)

        embed.title =           metadata.title
        embed.description =     metadata.channel
        embed.url =             metadata.url
        embed.timestamp =       interaction.created_at
        embed.color =           discord.Colour.from_rgb(r=255, g=0, b=0)

        return embed

