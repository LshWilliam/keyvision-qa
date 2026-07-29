from keyvision.data.schema import Annotation, ImageRecord
from keyvision.data.split import split_records


def _records(count: int = 30) -> list[ImageRecord]:
    return [
        ImageRecord(
            image=f"images/{index}.png",
            width=100,
            height=100,
            annotations=(Annotation((10, 10, 20, 20), index % 3, f"class_{index % 3}"),),
        )
        for index in range(count)
    ]


def test_split_is_reproducible_and_complete() -> None:
    first = split_records(_records(), seed=7)
    second = split_records(_records(), seed=7)
    assert first == second
    assert sum(len(values) for values in first.values()) == 30
    assert {record.image for values in first.values() for record in values} == {
        f"images/{index}.png" for index in range(30)
    }
