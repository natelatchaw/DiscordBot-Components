"""
Contains components related to OpenAI functionality.
"""

import logging
import textwrap
from logging import Logger
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from bot import Database, Settings
from bot.configuration import Section
from discord import Embed, Interaction, Member, User
from discord.app_commands import Choice, Range, choices, describe

import openai
from openai.openai_object import OpenAIObject

from models.openai import Submission

log: Logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE: Literal[2000] = 2000
"""
The maximum amount of characters
permitted in a Discord message
"""

# define model costs
MODEL_COSTS: Dict[str, float] = {
    'text-davinci-002': 0.0600 / 1000,
    'text-curie-001':   0.0060 / 1000,
    'text-babbage-001': 0.0012 / 1000,
    'text-ada-001':     0.0008 / 1000,
}

class OpenAI():
    """
    A collection of commands used to prompt the OpenAI GPT-3 AI models.
    AI model usage is tracked and can be calculated via the 'cost' command.
    """

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


    async def __send__(self, interaction: Interaction, prompt: str, model: str = 'text-davinci-002', tokens: Union[int, str] = 128, echo: bool = False) -> List[str]:
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

        # create submission object
        submission: Submission = Submission(interaction.id, user.id, model, prompt, '\n'.join(responses))
        # store the submission
        await self.__store__(interaction, submission=submission)

        # return the responses
        return responses


    async def __store__(self, interaction: Interaction, *, submission: Submission) -> None:
        # create the database
        self._database.create(Submission)
        # insert the submission
        self._database.insert(submission)


    async def __load__(self, interaction: Interaction) -> List[Submission]:
        # create the database
        self._database.create(Submission)
        # get all submissions
        submissions: List[Submission] = [Submission.__from_row__(row) for row in self._database.select(Submission)]
        # return submissions
        return submissions

    
    async def __print__(self, interaction: Interaction, *, responses: List[str]):
        # define the block tag for code block messages
        block_tag: str = '```'
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


    async def __get_cost__(self, submission: Submission, costs: Dict[str, float]) -> float:
        try:
            # retrieve the cost per token for the model used by the submission
            cost_per_token: float = costs[submission.model]
            # calculate the total cost
            total_cost: float = submission.token_count * cost_per_token
            # return the cost
            return total_cost
        except KeyError as error:
            log.warning(f'No cost defined for model {error}')
            return float(0)


    async def cost(self, interaction: Interaction) -> None:
        """
        Calculates an estimate of your OpenAI token usage.
        """
        
        # defer the interaction
        await interaction.response.defer(thinking=True)

        # load all submissions
        submissions: List[Submission] = await self.__load__(interaction)
        # get the message's author
        author: Union[User, Member] = interaction.user
        # get the target users
        users: List[Union[User, Member]] = [author]

        # for each user
        for user in users:
            # get all submissions by the user
            submissions = [submission for submission in submissions if submission.user_id == user.id]

            # initialize a dictionary
            per_model: Dict[str, List[Submission]] = dict()
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
                costs: List[float] = [await self.__get_cost__(submission, MODEL_COSTS) for submission in model_submissions]
                # add the cost data to the embed
                embed.add_field(name=model, value=f'${sum(costs):0.2f} ({len(costs)} submission{"s" if len(costs) != 1 else ""})')

            await interaction.followup.send(embed=embed)


    @describe(content='The input to provide to the AI model')
    @describe(model='The AI model to use for text generation')
    @choices(model=[
        Choice(name='Ada',      value='text-ada-001'),
        Choice(name='Babbage',  value='text-babbage-001'),
        Choice(name='Curie',    value='text-curie-001'),
        Choice(name='DaVinci',  value='text-davinci-002'),
    ])
    @describe(tokens='The maximum number of tokens to limit the response to')
    async def prompt(self, interaction: Interaction, content: str, model: str = 'text-davinci-002', tokens: Range[int, 64, 1024] = 128) -> None:
        """
        Provides a prompt to the designated AI model and generates a response.
        """

        try:
            # defer the interaction
            await interaction.response.defer(thinking=True)
            # send the prompt
            responses: List[str] = await self.__send__(interaction, prompt=content, model=model, tokens=tokens, echo=False)
            # send the responses
            await self.__print__(interaction, responses=responses)

        except Exception as error:
            await interaction.followup.send(f'{error}')


    @describe(a='A type of medium (e.g., book, poem, haiku, etc.)')
    @describe(about='The subject matter to generate the given medium from')
    @describe(model='The AI model to use for text generation')
    @choices(model=[
        Choice(name='Ada', value='text-ada-001'),
        Choice(name='Babbage', value='text-babbage-001'),
        Choice(name='Curie', value='text-curie-001'),
        Choice(name='DaVinci', value='text-davinci-002'),
    ])
    @describe(tokens='The maximum number of tokens to limit the response to')
    async def write(self, interaction: Interaction, a: str, about: str, model: str = 'text-davinci-002', tokens: Range[int, 64, 1024] = 128) -> None:
        """
        Generates a text response provided a prompt about what to write.
        """

        try:        
            # defer the interaction
            await interaction.response.defer(thinking=True)
            # initialize a list for content strings
            content: List[str] = list()
            # seed the prompt with the greentext prompt
            content.append('write')
            # append the value
            if a: content.append(f'a {a}')
            # append the value
            if about: content.append(f'about {about}')
            # join the content by spaces
            prompt: str = ' '.join(content)
            # send the prompt
            responses: List[str] = await self.__send__(interaction, prompt=prompt, model=model, tokens=tokens, echo=False)
            # send the responses
            await self.__print__(interaction, responses=responses)

        except Exception as error:
            await interaction.followup.send(f'{error}')
        

    @describe(be_me='A phrase to use for seeding the greentext (e.g., bottomless pit supervisor)')
    @describe(model='The AI model to use for text generation')
    @choices(model=[
        Choice(name='Ada', value='text-ada-001'),
        Choice(name='Babbage', value='text-babbage-001'),
        Choice(name='Curie', value='text-curie-001'),
        Choice(name='DaVinci', value='text-davinci-002'),
    ])
    @describe(tokens='The maximum number of tokens to limit the response to')
    async def greentext(self, interaction: Interaction, be_me: str, model: str = 'text-davinci-002', tokens: Range[int, 64, 1024] = 256) -> None:
        """
        Generates a 4chan-style greentext.
        """
        
        try:
            # defer the interaction
            await interaction.response.defer(thinking=True)
            # initialize a list for content strings
            content: List[str] = list()
            # seed the prompt with the greentext prompt
            content.append(textwrap.dedent('''
                generate a 4chan greentext

                >Be me
            '''))
            # if a be_me parameter was provided
            if be_me:
                # append the value
                content.append('>' + be_me)
            # join the content by newlines
            prompt: str = '\n'.join(content)
            # send the prompt
            responses: List[str] = await self.__send__(interaction, prompt=prompt, model=model, tokens=tokens, echo=True)
            # send the responses
            await self.__print__(interaction, responses=responses)
            
        except Exception as error:
            await interaction.followup.send(f'{error}')
