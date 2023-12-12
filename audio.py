from __future__ import annotations

import asyncio
import collections
import logging
import re
from asyncio import Event
from collections import Counter
from datetime import datetime
from logging import Logger
from pathlib import Path
from sqlite3 import Row
from typing import (Any, Dict, Generic, Iterable, Iterator, List, Optional,
                    Tuple, Type, TypeVar, Union)

import discord
import yt_dlp as youtube_dl
from bot import Database
from bot.configuration.section import Section
from bot.database.column import ColumnBuilder
from bot.database.storable import TStorable
from bot.database.table import Table, TableBuilder
from bot.settings.settings import Settings
from discord import AudioSource, Interaction
from discord.app_commands import Range, describe

RequestType = TypeVar('RequestType')

log: Logger = logging.getLogger(__name__)

class Audio():
    """
    Component responsible for audio playback.
    """

    #region Properties

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

    @property
    def before_options(self) -> Optional[str]:
        key: str = 'before_options'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ''
            return None
        
    @property
    def after_options(self) -> Optional[str]:
        key: str = 'after_options'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ''
            return None

    #endregion


    #region Lifecycle Events

    def __init__(self, *args, **kwargs):
        """
        Initializes state management objects for the audio loop
        """

        self._connection: Event = Event()
        self._playback_event: Event = Event()
        self._playback_queue: Audio.Queue[Audio.Request] = Audio.Queue()
        self._client: Optional[discord.VoiceClient] = None

        try:
            self._settings: Settings = kwargs['settings']
        except KeyError as error:
            raise Exception(f'Key {error} was not found in provided kwargs')


    async def __setup__(self) -> None:
        """
        Called after instance properties are initialized.
        """

        # create a config section for Audio
        self._settings.client[self.__class__.__name__] = Section(self.__class__.__name__, self._settings.client._reference, self._settings.client._parser)
        # create reference to Audio config section
        self._config: Section = self._settings.client[self.__class__.__name__]
        # create database instance
        self._database: Database = Database(Path(f'./data/{__name__}.db'))
        self._database.create(Audio.Metadata)
        # begin audio loop
        await self.__start__()

    #endregion


    #region Core Loop Events

    async def __start__(self):
        """
        The core audio playback loop.
        This is used internally and should not be called as a command.
        """

        while True:
            try:
                # wait for the connection event to be set
                await self._connection.wait()

                # if the voice client is not available
                if self._client is None:
                    # clear the connection event
                    self._connection.clear()
                    log.debug('Resetting; no voice client available')
                    # restart the loop
                    continue

                log.debug('Waiting for next audio playback request')
                # wait for the playback queue to return a request, or throw TimeoutError
                request: Audio.Request = await asyncio.wait_for(self._playback_queue.get(), self.timeout)

                # call dequeue hook logic
                await self.__on_dequeue__(request)

            except (asyncio.TimeoutError, Exception) as error:
                # call error hook logic
                await self.__on_error__(error)


    async def __on_dequeue__(self, request: Request) -> None:
        """
        Called when a request is retrieved from the queue.
        """

        # clear the playback event
        self._playback_event.clear()
        
        log.debug(f"Beginning track '{request.metadata.title}'")
        # return if client is unavailable
        if self._client is None: return

        # buffer out the first byte sequence of the source to prevent audio jitter
        start: bytes = request.source.read()
        # play the request
        self._client.play(request.source, after=self.__on_complete__)
        
        # wait for the playback event to be set
        await self._playback_event.wait()
        log.debug(f"Finishing track '{request.metadata.title}'")


    async def __on_error__(self, error: Exception) -> None:
        """
        Called when an error occurs while handling a request.
        """

        try:
            log.error(error)
            await self.__disconnect__()
        finally:
            # clear the connection event
            self._connection.clear()


    def __on_complete__(self, error: Optional[Exception]):
        """
        Called when a request completes in the audio loop.
        """

        # log error if it was provided
        if error: log.error(error)
        # set the playback event
        self._playback_event.set()

    #endregion


    #region Voice Client Management

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
                raise Audio.InvalidChannelError(None)

            # if the voice state does not reference a channel, raise error
            if not state.channel:
                raise Audio.InvalidChannelError(state.channel)

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


    async def __skip__(self, interaction: Interaction) -> Optional[Metadata]:
        """
        Skips the current track.
        """

        # store reference to the current metadata
        skipped: Optional[Audio.Request] = self._playback_queue.current
        # stop the voice client if available
        if self._client: self._client.stop()
        # return the skipped metadata
        return skipped.metadata if skipped else None


    async def __pause__(self, interaction: Interaction) -> None:
        """
        Pauses the current track.
        """
        # pause the voice client if available
        if self._client: self._client.pause()
        #
        return

    
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

    #endregion


    #region Business Logic

    async def __search__(self, interaction: Interaction, content: str, *, downloader: Optional[youtube_dl.YoutubeDL] = None) -> Optional[Metadata]:
        """
        Searches for video content and extracts the metadata.

        Parameters:
            - content: A string or URL to download metadata from
            - downloader: YoutubeDL downloader instance
        """
        
        # use provided downloader or initialize one if not provided
        downloader = downloader if downloader else youtube_dl.YoutubeDL(Audio.DEFAULTS)
        # extract info for the provided content
        data: Optional[Dict[str, Any]] = downloader.extract_info(content, download=False)

        # get the entries property, if it exists
        entries: Optional[List[Any]] = data.get('entries') if data else None
        # if the data contains a list of entries, use the list;
        # otherwise create list from data (single entry)
        results: List[Optional[Dict[str, Any]]] = entries if entries else [data]
        # return the first available result
        result: Optional[Dict[str, Any]] = results[0]

        # return a Metadata object if result exists
        return Audio.Metadata.__from_dict__(interaction, result) if result else None


    async def __queue__(self, interaction: Interaction, metadata: Metadata, *, before_options: List[str] = list(), after_options: List[str] = list()) -> Request:
        """
        Adds metadata to the queue.
        """

        # append any options stored in configuration
        if self.before_options:
            before_options.append(self.before_options)
        if self.after_options:
            after_options.append(self.after_options)

        # concatenate the options by space delimiter
        before: str = ' '.join(before_options)
        after: str = ' '.join(after_options)

        # create source from metadata url and options
        source: discord.AudioSource = discord.FFmpegOpusAudio(metadata.source, before_options=before, options=after)
        # create request from source and metadata
        request: Audio.Request = Audio.Request(source, metadata)

        # add the request to the queue
        await self._playback_queue.put(request)
        # insert the request to the database
        self._database.insert(Audio.Metadata, request.metadata)
        # return the request
        return request

    #endregion


    #region Application Commands

    @describe(content='A URL or video title to search for')
    @describe(speed='A numerical modifier to alter the audio speed by')
    async def play(self, interaction: Interaction, content: str, speed: Range[float, 0.5, 2.0] = 1.0) -> None:
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
            # download metadata for the provided content
            metadata: Optional[Audio.Metadata] = await self.__search__(interaction, content)
            if not metadata: raise Exception(f"No result found for '{content}'.")

            # get multiplier if speed was provided
            multiplier: Optional[float] = float(speed) if speed else None
            # assert the multiplier is within supported bounds
            multiplier = multiplier if multiplier and multiplier > 0.5 and multiplier < 2.0 else None
            # if a multiplier was specified, add it to the options list
            if multiplier and multiplier != 1.0: options.append(rf'-filter:a "atempo={multiplier}"')

            # queue the song and get the song's request data
            request: Optional[Audio.Request] = await self.__queue__(interaction, metadata, after_options=options)

            # generate an embed from the song request data
            embed: discord.Embed = Audio.RequestEmbed(interaction, request.metadata)

            # send the embed
            await followup.send(embed=embed)

        except youtube_dl.utils.DownloadError as exception:
            ansi_escape: re.Pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            inner: str = ansi_escape.sub('', str(exception.msg))
            await followup.send(f'An error occurred during download.\nDetails: {inner}')
            raise

        except Exception as exception:
            await followup.send(f'{exception}')
            raise

    async def queue(self, interaction: discord.Interaction) -> None:
        """
        Displays the request queue
        """

        followup: discord.Webhook = interaction.followup
        await interaction.response.defer(ephemeral=False, thinking=True)

        queue: List[Audio.Metadata] = [request.metadata for request in list(self._playback_queue)][:5]
        current: Optional[Audio.Metadata] = self._playback_queue.current.metadata if self._playback_queue.current else None
        
        # generate an embed from the song request data
        embed: Optional[discord.Embed] = Audio.RequestQueueEmbed(interaction, queue, current)
        
        # send the embed
        await followup.send(embed=embed)

    async def skip(self, interaction: discord.Interaction) -> None:
        """
        Skips the currently playing song
        """

        followup: discord.Webhook = interaction.followup
        await interaction.response.defer(ephemeral=False, thinking=True)

        # skip the song and get the skipped song's metadata.
        metadata: Optional[Audio.Metadata] = await self.__skip__(interaction)

        # generate an embed from the song request data
        embed: Optional[discord.Embed] = Audio.RequestEmbed(interaction, metadata) if metadata else None
        # determine whether the message should be ephemeral
        ephemeral: bool = not metadata
        # send the embed
        await followup.send(f'Skipped {metadata.title}' if metadata else 'Nothing is playing', ephemeral=ephemeral)

    async def top(self, interaction: discord.Interaction) -> None:
        """
        Displays your most frequent song requests
        """

        followup: discord.Webhook = interaction.followup
        await interaction.response.defer(ephemeral=False, thinking=True)

        # get the user's ID
        user_id: int = interaction.user.id
        # get the user's stored metadata
        results: List[Audio.Metadata] = [metadata for metadata in self._database.select(Audio.Metadata) if metadata.user_id == user_id]

        videos: Dict[str, Audio.Metadata] = { metadata.video_id : metadata for metadata in results }
        counted = Counter(metadata.video_id for metadata in results)
        top: List[Tuple[str, int]]= counted.most_common(5)
        output: List[Tuple[Audio.Metadata, int]] = [(videos[entry], count) for entry, count in top]

        # generate an embed from the song request data
        embed: Optional[discord.Embed] = Audio.RequestFrequencyEmbed(interaction, output)
        # send the embed
        await followup.send(embed=embed)



    #endregion


    #region Associated Classes

    class Metadata():

        @property
        def id(self) -> int:
            return self._id

        @property
        def user_id(self) -> int:
            return self._user_id

        @property
        def video_id(self) -> str:
            return self._video_id

        @property
        def title(self) -> str:
            return self._title

        @property
        def channel(self) -> str:
            return self._channel

        @property
        def thumbnail(self) -> str:
            return self._thumbnail

        @property
        def source(self) -> str:
            return self._source

        @property
        def url(self) -> str:
            return self._url


        def __init__(self, id: int, user_id: int, video_id: str, title: str, channel: str, thumbnail: str, url: str, source: str) -> None:
            self._id: int = id
            self._user_id: int = user_id
            self._video_id: str = video_id
            self._title: str = title
            self._channel: str = channel
            self._thumbnail: str = thumbnail
            self._url: str = url
            self._source: str = source

        def __str__(self) -> str:
            return f'[{self._video_id}] {self._title} ({self._url})'

        @classmethod
        def __from_dict__(cls, interaction: discord.Interaction, dict: Dict[str, Any]) -> Audio.Metadata:

            id: int = interaction.id
            user_id: int = interaction.user.id

            video_id: str = dict.get('id', str())
            if not isinstance(video_id, str): raise KeyError('id')

            title: str = dict.get('title', str())
            if not isinstance(title, str): raise KeyError('title')

            channel: str = dict.get('channel', str())
            if not isinstance(channel, str): raise KeyError('channel')

            thumbnail: str = dict.get('thumbnail', str())
            if not isinstance(channel, str): raise KeyError('thumbnail')

            url: str = dict.get('webpage_url', str())
            if not isinstance(url, str): raise KeyError('webpage_url')

            source: str = dict.get('url', str())
            if not isinstance(source, str): raise KeyError('url')

            return Audio.Metadata(id, user_id, video_id, title, channel, thumbnail, url, source)

        @classmethod
        def __table__(cls) -> Table:
            # create a table builder
            t_builder: TableBuilder = TableBuilder()
            # set the table's name
            t_builder.setName('Metadata')

            # create a column builder
            c_builder: ColumnBuilder = ColumnBuilder()
            # create timestamp column
            t_builder.addColumn(c_builder.setName('ID').setType('INTEGER').isPrimary().isUnique().column())
            # create user ID column
            t_builder.addColumn(c_builder.setName('UserID').setType('INTEGER').column())
            # create video ID column
            t_builder.addColumn(c_builder.setName('VideoID').setType('TEXT').column())
            # create title column
            t_builder.addColumn(c_builder.setName('Title').setType('TEXT').column())
            # create channel column
            t_builder.addColumn(c_builder.setName('Channel').setType('TEXT').column())
            # create channel column
            t_builder.addColumn(c_builder.setName('Thumbnail').setType('TEXT').column())
            # create url column
            t_builder.addColumn(c_builder.setName('URL').setType('TEXT').column())
            # create source column
            t_builder.addColumn(c_builder.setName('Source').setType('TEXT').column())

            # build the table
            table: Table = t_builder.table()
            # return the table
            return table

        def __values__(self) -> Tuple[Any, ...]:
            # create a tuple with the corresponding values
            value: Tuple[Any, ...] = (self.id, self.user_id, self.video_id, self.title, self.channel, self.thumbnail, self.url, self.source)
            # return the tuple
            return value

        @classmethod
        def __from_row__(cls: Type[TStorable], row: Row) -> Audio.Metadata:
            # Get ID value from the row
            id: int = row['ID']
            # Get UserID value from the row
            user_id: int = row['UserID']
            # Get VideoID value from the row
            video_id: str = row['VideoID']
            # Get Title value from the row
            title: str = row['Title']
            # Get Channel value from the row
            channel: str = row['Channel']
            # Get thumbnail value from the row
            thumbnail: str = row['Thumbnail']
            # Get thumbnail value from the row
            url: str = row['URL']
            # Get source value from the row
            source: str = row['Source']
            # return the Metadata
            return Audio.Metadata(id, user_id, video_id, title, channel, thumbnail, url, source)


    class Request():
        """
        A request object for the module to play
        """

        @property
        def metadata(self) -> Audio.Metadata:
            return self._metadata

        @property
        def source(self) -> AudioSource:
            return self._source


        def __init__(self, source: AudioSource, metadata: Audio.Metadata):
            self._metadata: Audio.Metadata = metadata
            self._source: AudioSource = source


    class Queue(Generic[RequestType], Iterable[RequestType]):

        @property
        def current(self) -> Optional[RequestType]:
            return self._current

        def __init__(self) -> None:
            self._queue: asyncio.Queue[RequestType] = asyncio.Queue()
            self._deque: collections.deque[RequestType] = collections.deque()
            self._current: Optional[RequestType] = None
            super().__init__()

        async def put(self, item: RequestType) -> None:
            await self._queue.put(item)
            self._deque.append(item)

        async def get(self) -> RequestType:
            self._current = None
            item: RequestType = await self._queue.get()
            self._current = self._deque.popleft()
            return item

        def __iter__(self) -> Iterator[RequestType]:
            return self._deque.__iter__()

    
    class RequestEmbed(discord.Embed):
        
        def __init__(self, interaction: discord.Interaction, current: Audio.Metadata, large_image: bool = True):
            color: discord.Color = discord.Color.blurple()
            user: Union[discord.User, discord.Member] = interaction.user
            title: str = current.title
            description: str = current.channel
            url: Optional[str] = current.url
            timestamp: Optional[datetime] = interaction.created_at
            super().__init__(color=color, title=title, description=description, url=url, timestamp=timestamp)
            self.set_author(name=user.display_name, icon_url=user.avatar.url if user.avatar else None)
            self.set_image(url=current.thumbnail if current and large_image else None)
            self.set_thumbnail(url=current.thumbnail if current and not large_image else None)

    
    class RequestQueueEmbed(discord.Embed):

        def __init__(self, interaction: discord.Interaction, metadata: List[Audio.Metadata], current: Optional[Audio.Metadata] = None, large_image: bool = False):
            color: discord.Color = discord.Color.blurple()
            user: Union[discord.User, discord.Member] = interaction.user
            title: str = current.title if current else 'Request Queue'
            description: Optional[str] = current.channel if current else None
            url: Optional[str] = None
            timestamp: Optional[datetime] = interaction.created_at
            super().__init__(color=color, title=title, description=description, url=url, timestamp=timestamp)
            self.set_author(name=user.display_name, icon_url=user.avatar.url if user.avatar else None)
            for item in metadata: self.add_field(name=item.title, value=item.channel, inline=False)
            self.set_image(url=current.thumbnail if current and large_image else None)
            self.set_thumbnail(url=current.thumbnail if current and not large_image else None)


    class RequestFrequencyEmbed(discord.Embed):

        def __init__(self, interaction: discord.Interaction, metadata: List[Tuple[Audio.Metadata, int]]):
            color: discord.Color = discord.Color.blurple()
            user: Union[discord.User, discord.Member] = interaction.user
            title: str = 'Top Requests'
            description: Optional[str] = None
            url: Optional[str] = None
            timestamp: Optional[datetime] = interaction.created_at
            super().__init__(color=color, title=title, description=description, url=url, timestamp=timestamp)
            self.set_author(name=user.display_name, icon_url=user.avatar.url if user.avatar else None)
            for item, count in metadata: self.add_field(name=item.title, value=f'{count} requests', inline=False)


    class Logger():
        def debug(self, message: str):
            log.debug(message)

        def warning(self, message: str):
            log.warning(message)

        def error(self, message: str):
            log.error(message)

    #endregion


    #region Error Classes

    class AudioError(Exception):
        """
        """

        def __init__(self, message: str, exception: Optional[Exception] = None):
            self._message = message
            self._inner_exception = exception

        def __str__(self) -> str:
            return self._message


    class NotConnectedError(AudioError):
        """
        """

        def __init__(self, exception: Optional[Exception] = None):
            message: str = f'The client is not connected to a compatible voice channel.'
            super().__init__(message, exception)


    class InvalidChannelError(AudioError):
        """
        """

        def __init__(self, channel: Optional[discord.abc.GuildChannel], exception: Optional[Exception] = None):
            reference: str = channel.mention if channel else 'unknown'
            message: str = f'Cannot connect to {reference} channel'
            super().__init__(message, exception)
    
    #endregion


    #region Constants

    AUDIO_LOGGER: Audio.Logger = Logger()

    DEFAULTS: Dict[str, Any] = {
        'default_search': 'auto',
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

    #endregion