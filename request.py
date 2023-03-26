import asyncio
import collections
from typing import Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar('T')

class RequestQueue(Generic[T], Iterable[T]):

    @property
    def current(self) -> Optional[T]:
        return self._current

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._deque: collections.deque[T] = collections.deque()
        self._current: Optional[T] = None
        super().__init__()

    async def put(self, item: T) -> None:
        await self._queue.put(item)
        self._deque.append(item)

    async def get(self) -> T:
        self._current = None
        item: T = await self._queue.get()
        self._current = self._deque.popleft()
        return item
    
    def __iter__(self) -> Iterator[T]:
        return self._deque.__iter__()