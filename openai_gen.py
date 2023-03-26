"""
Contains components related to OpenAI functionality.
"""

from __future__ import annotations

import logging
import re
import textwrap
from datetime import datetime, timedelta, timezone
from logging import Logger
from math import ceil
from pathlib import Path
from sqlite3 import Row
from typing import (Any, Dict, Iterable, List, Literal, Optional, Tuple, Type,
                    Union)

import openai
from bot import Database, Settings
from bot.configuration import Section
from bot.database.column import ColumnBuilder
from bot.database.storable import Storable
from bot.database.table import Table, TableBuilder
from discord import (Embed, Interaction,
                     Member, User)
from discord.app_commands import Choice, Range, choices, describe
from openai.openai_object import OpenAIObject

log: Logger = logging.getLogger(__name__)

class OpenAI():
    """
    A collection of commands used to prompt the OpenAI GPT-3 AI models.
    AI model usage is tracked and can be calculated via the 'cost' command.
    """

    #region Properties

    @property
    def key(self) -> Optional[str]:
        key: str = 'key'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ""
            return None
        
    @property
    def memory(self) -> Optional[int]:
        key: str = "memory"
        value: Optional[str] = None
        try:
            value = self._config[key]
            return int(value) if value else None
        except KeyError:
            self._config[key] = ""
            return None
        except ValueError:
            self._config[key] = ""
            return None
        
    @property
    def identity(self) -> Optional[str]:
        key: str = "identity"
        value: Optional[str] = None
        try:
            value = self._config[key]
            return value
        except KeyError:
            self._config[key] = ""
            return None
        except ValueError:
            self._config[key] = ""
            return None
    @identity.setter
    def identity(self, value: str) -> None:
        key: str = "identity"
        self._config[key] = value

    @property
    def is_enabled(self) -> bool:
        key: str = "enabled"
        value: Optional[str] = None
        try:
            value = self._config[key]
        except KeyError:
            self._config[key] = ""

        if value and isinstance(value, str):
            return value.lower() == str(True).lower()
        else:
            return False
    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        key: str = "enabled"
        self._config[key] = str(value)

    #endregion


    #region Lifecycle Events

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes and retrieves objects provided through args/kwargs
        """

        try:
            self._settings: Settings = kwargs['settings']
        except KeyError as error:
            raise Exception(f'Key {error} was not found in provided kwargs')


    async def __setup__(self, *args, **kwargs) -> None:
        key: str = self.__class__.__name__
        # create a config section for Audio
        self._settings.client[key] = Section(key, self._settings.client._parser, self._settings.client._reference)
        # create reference to Audio config section
        self._config: Section = self._settings.client[key]
        # set the api key
        openai.api_key = self.key
        # create database instance
        self._database: Database = Database(Path('./data/openai.db'))
        self._database.create(OpenAI.Submission)

        self._chats: Database = Database(Path('./data/chat.db'))
        self._chats.create(OpenAI.Chat)

    #endregion


    #region Business Logic

    async def __get_cost__(self, submission: Submission, costs: Dict[str, float]) -> float:
        try:
            # retrieve the cost per token for the model used by the submission
            cost_per_token: float = costs[submission.model]
            # calculate the total cost
            total_cost: float = submission.rate * submission.count
            # return the cost
            return total_cost
        except KeyError as error:
            log.warning(f'No cost defined for model {error}')
            return float(0)
        
        
    async def __send_completion__(self, interaction: Interaction, prompt: str, model: str = 'text-davinci-003', tokens: Union[int, str] = 128, echo: bool = False) -> List[str]:
        # check enabled configuration parameter
        if not self.is_enabled: raise ValueError('This command has been disabled.')
        # get the message's author
        user: Union[User, Member] = interaction.user
        # hash an ID from the author's ID
        id: int = hash(user.id)
        # convert the tokens parameter to an int if not already an int
        tokens = tokens if isinstance(tokens, int) else int(tokens)
        # create the Completion request
        completion: OpenAIObject = openai.Completion.create(model=model, prompt=prompt, max_tokens=tokens, echo=echo, user=str(id))  # type: ignore

        # get the list of choices
        choices: List[OpenAIObject] = completion.choices
        # get the list of text responses
        responses: List[str] = [choice.text for choice in choices]

        # get rate for the selected model
        rate: float = OpenAI.MODEL_COSTS[model]
        # create submission object
        submission: OpenAI.Submission = OpenAI.TextSubmission(interaction.id, user.id, rate, model, prompt, '\n'.join(responses))
        # store the submission
        self._database.insert(OpenAI.Submission, submission)

        # return the responses
        return responses


    async def __send_image__(self, interaction: Interaction, prompt: str, size: str = '512x512', count: Union[int, str] = 1) -> List[str]:
        # check enabled configuration parameter
        if not self.is_enabled: raise ValueError('This command has been disabled.')
        # get the message's author
        user: Union[User, Member] = interaction.user
        # hash an ID from the author's ID
        id: int = hash(user.id)
        # convert the count parameter to an int if not already an int
        count = count if isinstance(count, int) else int(count)
        # create the Image request
        response: OpenAIObject = openai.Image.create(prompt=prompt, n=count, size=size, response_format='url', user=str(id))  # type: ignore

        # get the data list
        images: List[OpenAIObject] = response.data
        # get the list of image references
        responses: List[str] = [image['url'] for image in images]

        # get rate for the selected model
        rate: float = OpenAI.MODEL_COSTS[f'image-{size}']
        # create submission object
        submission: OpenAI.Submission = OpenAI.ImageSubmission(interaction.id, user.id, rate, count, size)
        # store the submission
        self._database.insert(OpenAI.Submission, submission)

        return responses
        
    
    async def __print__(self, interaction: Interaction, *, responses: List[str], block_tag: str = '```'):
        # for each returned response
        for response in responses:
            # calculate the max characters that can be inserted into each message's code block
            max_characters: int = MAX_MESSAGE_SIZE - (2 * len(block_tag))
            # break the response into string segments with length equivalent to the maximum character limit
            segments: List[str] = textwrap.wrap(response, max_characters, break_long_words=False, replace_whitespace=False)
            # for each segment
            for segment in segments:
                # send the segment as a followup message
                await interaction.followup.send(f'{block_tag}\n{segment}\n{block_tag}')

    #endregion


    #region Application Commands

    async def cost(self, interaction: Interaction) -> None:
        """
        Calculates an estimate of your OpenAI token usage.
        """
        
        # defer the interaction
        await interaction.response.defer(thinking=True)

        # load all submissions
        submissions: Iterable[OpenAI.Submission] = self._database.select(OpenAI.Submission)
        # get the message's author
        author: Union[User, Member] = interaction.user
        # get the target users
        users: List[Union[User, Member]] = [author]

        # for each user
        for user in users:
            # get all submissions by the user
            submissions = [submission for submission in submissions if submission.user_id == user.id]

            # initialize a dictionary
            per_model: Dict[str, List[OpenAI.Submission]] = dict()
            # for each submission
            for submission in submissions:
                # if the per_model dictionary does not have the model as a key
                if not per_model.get(submission.model):
                    # add the model as a key with a list
                    per_model[submission.model] = list()
                # append the submission to the list for the submission's model
                per_model[submission.model].append(submission)

            embed: Embed = Embed()
            embed.title = 'OpenAI Usage'
            embed.description = f'{len(submissions)} Total Submission{"s" if len(submissions) != 1 else ""}'
            embed.set_author(name=user.name, icon_url=user.avatar.url if user.avatar else None)

            # for each entry in per_model
            for model, model_submissions in per_model.items():
                # calculate the cost for each submission
                costs: List[float] = [await self.__get_cost__(submission, OpenAI.MODEL_COSTS) for submission in model_submissions]
                # add the cost data to the embed
                embed.add_field(name=model, value=f'${sum(costs):0.2f} ({len(costs)} submission{"s" if len(costs) != 1 else ""})')

            await interaction.followup.send(embed=embed)


    @describe(prompt='The prompt to use for the GPT model identity')
    async def set_identity(self, interaction: Interaction, prompt: str) -> None:
        """
        Set the system identity prompt for the GPT model
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        self.identity = prompt
        flavor: str = 'Got it. I will now try to adhere to the following identity:'

        #
        await interaction.followup.send(f'{flavor}\n{self.identity}')


    @describe(message='The message to send to the GPT model')
    @describe(model='The GPT model to chat with')
    @choices(model=[
        Choice(name='ChatGPT',  value='gpt-3.5-turbo'),
        #Choice(name='DaVinci',  value='text-davinci-003'),
        #Choice(name='Curie',    value='text-curie-001'),
        #Choice(name='Babbage',  value='text-babbage-001'),
        #Choice(name='Ada',      value='text-ada-001'),
    ])
    async def chat(self, interaction: Interaction, message: str, model: str = 'gpt-3.5-turbo') -> None:
        """
        Chat with a GPT model.
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        channel_id: Optional[int] = interaction.channel.id if interaction.channel else None

        if not channel_id: raise Exception(f'Could not determine channel ID.')

        # establish cutoff timestamp
        cutoff: datetime = interaction.created_at - timedelta(hours=3)

        # get chat messages from database
        history: Iterable[OpenAI.Chat] = [chat for chat in self._chats.select(OpenAI.Chat)]
        # filter chat messages to messages in user's thread
        history = filter(lambda chat: chat.thread_id == channel_id, history)
        # sort chat messages by timestamp
        history = sorted(history, key=lambda chat: chat.timestamp, reverse=False)

        # get setting for memory
        memory: int = self.memory if self.memory else 10
        # take the most recent messages, specified by the memory settings
        history = list(history)[-1 * memory:]

        log.debug(f'Including {len(list(history))} messages from chat history')

        # transform chats to dict format
        messages: List[Dict[str, str]] = [chat.to_dict() for chat in history]
        # create query chat message from interaction
        prompt_chat: OpenAI.Chat = OpenAI.Chat(interaction.created_at, interaction.user.id, channel_id, message, 0)
        # convert the chat to a dict
        prompt: Dict[str, str] = prompt_chat.to_dict()
        # add the converted chat to the array of messages
        messages.append(prompt)

        identity: Optional[Dict[str, str]] = { 'role': 'system', 'content': self.identity } if self.identity else None
        # insert the identity prompt
        if identity: messages.insert(0, identity)

        # create a chat completion
        response: Dict[str, Any] = openai.ChatCompletion.create(
            model=model,
            messages = messages
        ) # type: ignore

        # retreive request usage data
        usage: Dict[str, int] = response['usage']
        # retrieve prompt token usage
        prompt_tokens: int = usage['prompt_tokens']
        # update prompt chat's token count
        prompt_chat.tokens = prompt_tokens

        # convert the reply dict to a chat object
        completion_chat = OpenAI.Chat.from_response(response, interaction.user)
        
        # store the query chat message
        self._chats.insert(OpenAI.Chat, prompt_chat)
        # insert the reply chat into the database 
        self._chats.insert(OpenAI.Chat, completion_chat)

        await interaction.followup.send(message)
        await interaction.followup.send(completion_chat.content)


    @describe(prompt='The input to provide to the AI model')
    @describe(size='The size of the images to generate')
    @choices(size=[
        Choice(name='Small',    value='256x256'),
        Choice(name='Medium',   value='512x512'),
        Choice(name='Large',    value='1024x1024'),
    ])
    @describe(images='The number of images to generate')
    async def image(self, interaction: Interaction, prompt: str, size: str = '512x512', images: Range[int, 1, 2] = 1) -> None:
        """
        Provides a prompt to the AI model and generates image responses.
        """

        try:
            # defer the interaction
            await interaction.response.defer(thinking=True)
            # send the prompt
            responses: List[str] = await self.__send_image__(interaction, prompt=prompt, size=size, count=images)
            # send the responses
            await self.__print__(interaction, responses=responses, block_tag='')
        
        except Exception as error:
            await interaction.followup.send(f'{error}')

    #endregion


    #region Associated Classes

    class Chat(Storable):

        def __init__(self, timestamp: datetime, user_id: int, thread_id: int, content: str, tokens: int) -> None:
            self._timestamp: datetime = timestamp
            self._user_id: int = user_id
            self._thread_id: int = thread_id
            self._content: str = content
            self._tokens: int = tokens
        
        @property
        def timestamp(self) -> datetime:
            return self._timestamp
        @property
        def user_id(self) -> int:
            return self._user_id
        @property
        def thread_id(self) -> int:
            return self._thread_id
        @property
        def content(self) -> str:
            return self._content
        @property
        def tokens(self) -> int:
            return self._tokens
        @tokens.setter
        def tokens(self, value: int) -> None:
            self._tokens = value

        @classmethod
        def __table__(cls) -> Table:
            # create a table builder
            t_builder: TableBuilder = TableBuilder()
            t_builder.setName('Chats')

            # create a column builder
            c_builder: ColumnBuilder = ColumnBuilder()
            t_builder.addColumn(c_builder.setName('Timestamp').setType('TIMESTAMP').isPrimary().isUnique().column())
            t_builder.addColumn(c_builder.setName('UserID').setType('INTEGER').column())
            t_builder.addColumn(c_builder.setName('ThreadID').setType('INTEGER').column())
            t_builder.addColumn(c_builder.setName('Content').setType('TEXT').column())
            t_builder.addColumn(c_builder.setName('Tokens').setType('INTEGER').column())

            # build the table
            table: Table = t_builder.table()
            # return the table
            return table

        def __values__(self) -> Tuple[Any, ...]:
            # create a tuple with the corresponding values
            value: Tuple[Any, ...] = (self._timestamp, self._user_id, self._thread_id, self._content, self._tokens)
            # return the tuple
            return value

        @classmethod
        def __from_row__(cls: Type[OpenAI.Chat], row: Row) -> OpenAI.Chat:
            timestamp: datetime = row['Timestamp']
            user_id: int = row['UserID']
            thread_id: int = row['ThreadID']
            content: str = row['Content']
            tokens: int = row['Tokens']
            # return the Submission
            return cls(timestamp, user_id, thread_id, content, tokens)
        
        def to_dict(self) -> Dict[str, str]:
            # set role to assistant if user_id is 0, otherwise set role to user
            role: str = "assistant" if self.user_id == 0 else "user"
            #
            content: str = self.content
            # return data
            return { "role": role, "content": content }
        
        @classmethod
        def from_dict(cls: Type[OpenAI.Chat], dict: Dict[str, Any], user: Union[User, Member], *, tokens: int, timestamp: datetime):
            # get the user's ID
            user_id: int = user.id if dict['role'] == 'user' else 0
            # set the thread ID to the user's ID
            thread_id: int = user.id
            # get the message content
            content: str = dict['content']
            return cls(timestamp, user_id, thread_id, content.strip(), tokens)
        
        @classmethod
        def from_response(cls: Type[OpenAI.Chat], dict: Dict[str, Any], user: Union[User, Member]):
            # retrieve timestamp data
            created: int = dict['created']
            # create datetime object from timestamp
            timestamp: datetime = datetime.fromtimestamp(float(created), tz=timezone.utc)

            # retreive request usage data
            usage: Dict[str, int] = dict['usage']
            # retrieve total token usage
            tokens: int = usage['completion_tokens']

            # retrieve list of reply choices
            choices: List[Dict[str, Any]] = dict['choices']
            # select the first choice from list of choices
            choice: Dict[str, Any] = choices[0]
            # get the message object from the selected choice
            message: Dict[str, str] = choice['message']
            
            return cls.from_dict(message, user, tokens=tokens, timestamp=datetime.now(tz=timezone.utc))
            
        

    class Submission(Storable):

        def __init__(self, id: int, user_id: int, rate: float, count: int, model: str) -> None:
            self._id: int = id
            self._user_id: int = user_id
            self._rate: float = rate
            self._count: int = count
            self._model: str = model

        @property
        def id(self) -> int:
            return self._id

        @property
        def user_id(self) -> int:
            return self._user_id

        @property
        def rate(self) -> float:
            return self._rate

        @property
        def count(self) -> int:
            return self._count

        @property
        def model(self) -> str:
            return self._model

        @classmethod
        def __table__(cls) -> Table:
            # create a table builder
            t_builder: TableBuilder = TableBuilder()
            # set the table's name
            t_builder.setName('Submissions')

            # create a column builder
            c_builder: ColumnBuilder = ColumnBuilder()
            # create id column
            t_builder.addColumn(c_builder.setName('ID').setType('INTEGER').isPrimary().isUnique().column())
            # create user ID column
            t_builder.addColumn(c_builder.setName('UserID').setType('INTEGER').column())
            # create rate column
            t_builder.addColumn(c_builder.setName('Rate').setType('REAL').column())
            # create count column
            t_builder.addColumn(c_builder.setName('Count').setType('INTEGER').column())
            # create model column
            t_builder.addColumn(c_builder.setName('Model').setType('TEXT').column())

            # build the table
            table: Table = t_builder.table()
            # return the table
            return table

        def __values__(self) -> Tuple[Any, ...]:
            # create a tuple with the corresponding values
            value: Tuple[Any, ...] = (self.id, self.user_id, self.rate, self.count, self.model)
            # return the tuple
            return value

        @classmethod
        def __from_row__(cls: Type[OpenAI.Submission], row: Row) -> OpenAI.Submission:
            # get ID value from the row
            id: int = row['ID']
            # get UserID value from the row
            user_id: int = row['UserID']
            # get Rate value from the row
            rate: float = row['Rate']
            # get Count value from the row
            count: int = row['Count']
            # get Model value from the row
            model: str = row['Model']
            # return the Submission
            return OpenAI.Submission(id, user_id, rate, count, model)


    class ImageSubmission(Submission):

        def __init__(self, id: int, user_id: int, rate: float, count: int, size: str) -> None:
            self._id: int = id
            self._user_id: int = user_id
            self._rate: float = rate
            self._count: int = count
            self._model: str = f'image-{size}'

            super().__init__(self._id, self._user_id, self._rate, self._count, self._model)

        @property
        def id(self) -> int:
            return self._id

        @property
        def user_id(self) -> int:
            return self._user_id

        @property
        def rate(self) -> float:
            return self._rate

        @property
        def count(self) -> int:
            return self._count

        @property
        def model(self) -> str:
            return self._model



    class TextSubmission(Submission):

        def __init__(self, id: int, user_id: int, rate: float, prompt: str, response: str, model: str) -> None:
            self._id: int = id
            self._user_id: int = user_id
            self._rate: float = rate

            self._prompt: str = prompt
            self._response: str = response

            self._count: int = self.__get_tokens__()
            self._model: str = model 

            super().__init__(self._id, self._user_id, self._rate, self._count, self._model)

        @property
        def id(self) -> int:
            return self._id

        @property
        def user_id(self) -> int:
            return self._user_id

        @property
        def rate(self) -> float:
            return self._rate

        @property
        def count(self) -> int:
            return self._count

        @property
        def model(self) -> str:
            return self._model


        @property
        def prompt(self) -> str:
            return self._prompt

        @property
        def response(self) -> str:
            return self._response


        def __get_tokens__(self) -> int:
            # split the prompt by whitespace characters
            prompt_segments: List[str] = re.split(r"[\s]+", self._prompt)
            # get a token count for each word
            prompt_token_counts: List[int] = [ceil(len(prompt_segment) / 4) for prompt_segment in prompt_segments]

            # split the response by whitespace characters
            response_segments: List[str] = re.split(r"[\s]+", self._prompt)
            # get a token count for each word
            response_token_counts: List[int] = [ceil(len(response_segment) / 4) for response_segment in response_segments]

            # return the sum of token counts
            return sum(prompt_token_counts) + sum(response_token_counts)
        
    #endregion


    #region Constants

    # define model costs
    MODEL_COSTS: Dict[str, float] = {
        'text-davinci-003': 0.0600 / 1000,
        'text-davinci-002': 0.0600 / 1000,
        'text-curie-001':   0.0060 / 1000,
        'text-babbage-001': 0.0012 / 1000,
        'text-ada-001':     0.0008 / 1000,
        'image-1024x1024':  0.0200 / 1,
        'image-512x512':    0.0180 / 1,
        'image-256x256':    0.0160 / 1,
    }

    #endregion
    

MAX_MESSAGE_SIZE: Literal[2000] = 2000
"""
The maximum amount of characters
permitted in a Discord message
"""