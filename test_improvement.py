import time
from dataclasses import dataclass

from litestar.dto import DataclassDTO, DTOConfig
from litestar.typing import FieldDefinition


@dataclass
class LeafNode:
    name: str
    parent: "LeafNode | None"
    children: "list[LeafNode] | None"


# Test with increasing depths
for depth in [3, 5, 7, 9]:
    class TestDTO(DataclassDTO[LeafNode]):
        config = DTOConfig(
            experimental_codegen_backend=True,
            max_nested_depth=depth,
        )
    
    print(f"Testing with max_nested_depth={depth}:")
    start = time.perf_counter()
    
    TestDTO.create_for_field_definition(
        field_definition=FieldDefinition.from_annotation(LeafNode),
        handler_id=f"test-case-{depth}",
    )
    
    elapsed = time.perf_counter() - start
    print(f"  Time taken: {elapsed:.4f} seconds")
    
    # Clear the cache to test each depth independently
    from litestar.dto._codegen_backend import TransferFunctionFactory
    TransferFunctionFactory._transfer_function_cache.clear()