# DiscordBot-Components

## Introduction
The purpose of this repository is to abstract out concrete command implementations from the core [DiscordBot](https://github.com/natelatchaw/DiscordBot) implementation. Dependencies of commands are subdivided into individual *requirements.txt* files in the [/requirements](https://github.com/natelatchaw/DiscordBot-Components/tree/master/requirements) directory, which are in turn referenced by the main [requirements.txt](https://github.com/natelatchaw/DiscordBot-Components/blob/master/requirements.txt) for concise installation.

## Component Interface
Components are implemented as [Python classes](https://docs.python.org/3/tutorial/classes.html). Lifecycle hooks are implemented as magic/dunder methods and are optional. Unimplemented hooks are equivalent to a no-op at runtime. See [Lifecycle Hooks](https://github.com/natelatchaw/DiscordBot#Lifecycle-Hooks) for available hooks.

DiscordBot will attempt to register all async methods included as class members as application commands. If the methods has an improper [signature](#command-signature), command registration will be cancelled and the method will be passed over.

## Command Signature
Commands are class members, implemented as asynchronous methods. The method signature should be constructed as defined by the [Commands section](https://discordpy.readthedocs.io/en/latest/interactions/api.html#commands) of the discord.py documentation.

In general, the method should have an [discord.Interaction](https://discordpy.readthedocs.io/en/latest/interactions/api.html#interaction) object as the first positional parameter. This object is used to obtain metadata about the interaction such as the user that invoked the interaction, the timestamp representing when the interaction was invoked, among other details. Responding to the interaction is also done though methods associated with this object. See the [discord.py documentation](https://discordpy.readthedocs.io/en/latest) for further details.

Further positional and keyword arguments are used to add arguments to the application command. In general, supported argument types are limited to:
- [Text Sequence Types](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)
- [Numeric Types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)
- [discord.app_commands.Transformer](https://discordpy.readthedocs.io/en/latest/interactions/api.html#transformer)
- [discord.app_commands.Range](https://discordpy.readthedocs.io/en/latest/interactions/api.html#range)
- other applicable types provided by discord.app_commands
