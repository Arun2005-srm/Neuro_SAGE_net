from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SampleRecord:
    path: str
    label: int
    class_name: str
    sample_id: str
    group_id: str | None = None

    def to_dict(self):
        return asdict(self)
