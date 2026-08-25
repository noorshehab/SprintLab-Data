from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any,List,Optional
from collections import defaultdict

Request=dict[str, Any]

class Mediator(ABC):
    def request(self,request:Request)->Any:
        pass

class Component:  
    def __init__(self, mediator: Optional[Mediator] = None) -> None:
        self._mediator = mediator

    def set_mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator

    @property
    def mediator(self) -> Mediator:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator

class SigletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Publisher(ABC):
    def __init__(self)->None:
        self.subscribers=defaultdict(list)

    def subscribe(self,subscriber:Subscriber,subscriber_type:str)->None:
        pass

    def unsubscribe(self,subscriber:Subscriber,subscriber_type:str)->None:
        pass
              
    def notify(self,event_type:str,event_data:dict)->None:
        pass


class Subscriber(ABC):
    def update(self,event:str,event_data:dict)->None:
        pass
