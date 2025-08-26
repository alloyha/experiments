# Provide a python example comparison between dataclass, attrs and pydantic, timing their definitions

from dataclasses import dataclass
from attrs import define
from pydantic import BaseModel


@dataclass
class DataClass:
    name: str
    age: int


@define
class Attrs:
    name: str
    age: int


class Pydantic(BaseModel):
    name: str
    age: int


if __name__ == "__main__":
    import timeit

    data = {"name": "John Doe", "age": 30}

    # Dataclass
    dataclass_time = timeit.timeit(
        "DataClass(**data)", setup="from __main__ import DataClass, data", number=100000
    )

    # Attrs
    attrs_time = timeit.timeit(
        "Attrs(**data)", setup="from __main__ import Attrs, data", number=100000
    )

    # Pydantic
    pydantic_time = timeit.timeit(
        "Pydantic(**data)", setup="from __main__ import Pydantic, data", number=100000
    )

    # Results
    print(f"Dataclass: {dataclass_time:.6f} seconds")
    print(f"Attrs: {attrs_time:.6f} seconds")
    print(f"Pydantic: {pydantic_time:.6f} seconds")