import time
from dataclasses import dataclass

from litestar.dto import DataclassDTO, DTOConfig
from litestar.typing import FieldDefinition


@dataclass
class LeafNode:
    name: str

    parent: "LeafNode | None"
    children: "list[LeafNode] | None"


class LeafNodeViewDTO5(DataclassDTO[LeafNode]):
    config = DTOConfig(
        experimental_codegen_backend=True,
        max_nested_depth=5,
    )

print("Testing with max_nested_depth=5:")
start = time.perf_counter()

LeafNodeViewDTO5.create_for_field_definition(
    field_definition=FieldDefinition.from_annotation(LeafNode),
    handler_id="test-case-5",
)

elapsed_5 = time.perf_counter() - start
print(f"Time taken: {elapsed_5:.4f} seconds")

# Test with depth 7
class LeafNodeViewDTO7(DataclassDTO[LeafNode]):
    config = DTOConfig(
        experimental_codegen_backend=True,
        max_nested_depth=7,
    )

print("\nTesting with max_nested_depth=7:")
start = time.perf_counter()

LeafNodeViewDTO7.create_for_field_definition(
    field_definition=FieldDefinition.from_annotation(LeafNode),
    handler_id="test-case-7",
)

elapsed_7 = time.perf_counter() - start
print(f"Time taken: {elapsed_7:.4f} seconds")

# Test without codegen
class LeafNodeViewDTONoCodegen(DataclassDTO[LeafNode]):
    config = DTOConfig(
        experimental_codegen_backend=False,
        max_nested_depth=7,
    )

print("\nTesting with max_nested_depth=7 (no codegen):")
start = time.perf_counter()

LeafNodeViewDTONoCodegen.create_for_field_definition(
    field_definition=FieldDefinition.from_annotation(LeafNode),
    handler_id="test-case-no-codegen",
)

elapsed_no_codegen = time.perf_counter() - start
print(f"Time taken: {elapsed_no_codegen:.4f} seconds")