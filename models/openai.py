import re
from math import ceil
from sqlite3 import Row
from typing import Any, List, Tuple, Type

from database.column import ColumnBuilder
from database.storable import Storable, TStorable
from database.table import Table, TableBuilder


class Submission(Storable):

    def __init__(self, id: int, user_id: int, model: str, prompt: str, response: str) -> None:
        self._id: int = id
        self._user_id: int = user_id
        self._model: str = model
        self._prompt: str = prompt
        self._response: str = response
        self._token_count: int = self.__get_tokens__()

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

    @property
    def id(self) -> int:
        return self._id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def response(self) -> str:
        return self._response
    
    @property
    def token_count(self) -> int:
        return self._token_count

    @classmethod
    def __table__(self) -> Table:
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
        # create model column
        t_builder.addColumn(c_builder.setName('Model').setType('TEXT').column())
        # create prompt column
        t_builder.addColumn(c_builder.setName('Prompt').setType('TEXT').column())
        # create response column
        t_builder.addColumn(c_builder.setName('Response').setType('TEXT').column())
        # create token count column
        t_builder.addColumn(c_builder.setName('TokenCount').setType('INTEGER').column())
        
        # build the table
        table: Table = t_builder.table()
        # return the table
        return table

    def __values__(self) -> Tuple[Any, ...]:
        # create a tuple with the corresponding values
        value: Tuple[Any, ...] = (self.id, self.user_id, self.model, self.prompt, self.response, self.token_count)
        # return the tuple
        return value
        
    @classmethod
    def __from_row__(cls: Type[TStorable], row: Row) -> TStorable:
        # get ID value from the row
        id: int = row['ID']
        # get UserID value from the row
        user_id: int = row['UserID']
        # get Model value from the row
        model: str = row['Model']
        # get Prompt value from the row
        prompt: str = row['Prompt']
        # get Response value from the row
        response: str = row['Response']
        # get TokenCount value from the row
        token_count: int = row['TokenCount']
        # return the Submission
        return Submission(id, user_id, model, prompt, response)
