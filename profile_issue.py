import cProfile
import pstats
import io
from dataclasses import dataclass

from litestar.dto import DataclassDTO, DTOConfig
from litestar.typing import FieldDefinition


@dataclass
class LeafNode:
    name: str
    parent: "LeafNode | None"
    children: "list[LeafNode] | None"


class LeafNodeViewDTO(DataclassDTO[LeafNode]):
    config = DTOConfig(
        experimental_codegen_backend=True,
        max_nested_depth=7,
    )


pr = cProfile.Profile()
pr.enable()

LeafNodeViewDTO.create_for_field_definition(
    field_definition=FieldDefinition.from_annotation(LeafNode),
    handler_id="test-case",
)

pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Print top 30 functions

print(s.getvalue())