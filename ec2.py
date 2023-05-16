import logging
from datetime import datetime
from logging import Logger
from typing import Any, Dict, List, Optional

import boto3
from bot import Settings
from bot.configuration import Section
from botocore.exceptions import ClientError
from discord import Embed, Interaction

log: Logger = logging.getLogger(__name__)

class EC2():
    """
    A collection of commands used to manage AWS EC2 instances.
    """

    @property
    def region(self) -> Optional[str]:
        key: str = 'region'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ""
            return None
        
    @property
    def key_id(self) -> Optional[str]:
        key: str = 'access_key_id'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ""
            return None
        
    @property
    def key_value(self) -> Optional[str]:
        key: str = 'secret_access_key'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ""
            return None
        
    @property
    def instance_id(self) -> Optional[str]:
        key: str = 'instance_id'
        value: Optional[str] = None
        try:
            value = self._config[key]
            if value and isinstance(value, str):
                return value
        except:
            self._config[key] = ""
            return None
        
    

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

        params: Dict[str, str] = {
            'aws_access_key_id': self.key_id,
            'aws_secret_access_key': self.key_value,
            'region_name': self.region
        }
        
        self.ec2 = boto3.client('ec2', **params)

    async def info(self, interaction: Interaction) -> None:
        """
        Gets info about the preconfigured EC2 instance
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        if not self.instance_id:
            await interaction.followup.send("No instance ID was provided.")
            return
        
        try:
            response: List[Any] = self.ec2.describe_instances(InstanceIds=[self.instance_id], DryRun=False)
        except ClientError as error:
            await interaction.followup.send("An error occurred. Check the logs for details.")
            log.error(error)
            return

        reservation = response['Reservations'][0]
        instance = reservation['Instances'][0]

        state: str = instance['State']['Name']
        ip: str = instance['PublicIpAddress']
        launch: datetime = instance['LaunchTime']
        type: str = instance['InstanceType']

        embed: Embed = Embed()
        embed.title = "EC2 Instance"
        embed.add_field(name='IP', value=ip, inline=False)
        embed.add_field(name='Status', value=state, inline=False)
        embed.add_field(name='Size', value=type, inline=False)
        embed.timestamp = launch

        await interaction.followup.send(embed=embed)


    async def start(self, interaction: Interaction) -> None:
        """
        Starts the preconfigured EC2 instance
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        if not self.instance_id:
            await interaction.followup.send("No instance ID was provided.")
            return
        
        try:
            self.ec2.start_instances(InstanceIds=[self.instance_id], DryRun=False)
        except ClientError as error:
            await interaction.followup.send("An error occurred. Check the logs for details.")
            log.error(error)

        #
        await interaction.followup.send(f'EC2 instance started successfully.')


    async def stop(self, interaction: Interaction) -> None:
        """
        Stops the preconfigured EC2 instance
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        if not self.instance_id:
            await interaction.followup.send("No instance ID was provided.")
            return
        
        try:
            self.ec2.stop_instances(InstanceIds=[self.instance_id], DryRun=False)
        except ClientError as error:
            await interaction.followup.send("An error occurred. Check the logs for details.")
            log.error(error)

        #
        await interaction.followup.send(f'EC2 instance stopped successfully.')


    async def restart(self, interaction: Interaction) -> None:
        """
        Restarts the preconfigured EC2 instance
        """

        # defer the interaction
        await interaction.response.defer(thinking=True, ephemeral=False)

        if not self.instance_id:
            await interaction.followup.send("No instance ID was provided.")
            return
        
        try:
            self.ec2.reboot_instances(InstanceIds=[self.instance_id], DryRun=False)
        except ClientError as error:
            await interaction.followup.send("An error occurred. Check the logs for details.")
            log.error(error)

        #
        await interaction.followup.send(f'EC2 instance restarted successfully.')