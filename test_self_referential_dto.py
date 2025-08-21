"""Test self-referential DTO optimization."""
import time
from dataclasses import dataclass
from typing import Any

import pytest

from litestar.dto import DataclassDTO, DTOConfig
from litestar.typing import FieldDefinition


@dataclass
class TreeNode:
    """A self-referential tree structure."""
    name: str
    parent: "TreeNode | None" = None
    children: "list[TreeNode] | None" = None


@dataclass
class LinkedListNode:
    """A self-referential linked list."""
    value: int
    next: "LinkedListNode | None" = None


def test_self_referential_dto_performance():
    """Test that self-referential DTOs don't have exponential time complexity."""
    times = []
    
    for depth in [5, 6, 7]:
        class TestDTO(DataclassDTO[TreeNode]):
            config = DTOConfig(
                experimental_codegen_backend=True,
                max_nested_depth=depth,
            )
        
        start = time.perf_counter()
        TestDTO.create_for_field_definition(
            field_definition=FieldDefinition.from_annotation(TreeNode),
            handler_id=f"test-{depth}",
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        # Clear cache for next iteration
        from litestar.dto._codegen_backend import TransferFunctionFactory
        TransferFunctionFactory._transfer_function_cache.clear()
    
    # Check that time doesn't grow exponentially
    # If optimization works, time should grow roughly linearly
    # Without optimization, time[2] would be ~5x time[1]
    growth_factor = times[2] / times[1] if times[1] > 0 else float('inf')
    assert growth_factor < 3.0, f"Time growing too fast: {times}"
    print(f"Times: {times}, Growth factor: {growth_factor:.2f}")


def test_self_referential_dto_correctness():
    """Test that self-referential DTOs work correctly."""
    
    class TreeNodeDTO(DataclassDTO[TreeNode]):
        config = DTOConfig(
            experimental_codegen_backend=True,
            max_nested_depth=3,
        )
    
    TreeNodeDTO.create_for_field_definition(
        field_definition=FieldDefinition.from_annotation(TreeNode),
        handler_id="test-correctness",
    )
    
    # Get the backend from the DTO class
    backend = TreeNodeDTO._dto_backends["test-correctness"]["return_backend"]
    
    # Create a simple tree structure
    tree_data = {
        "name": "root",
        "parent": None,
        "children": [
            {
                "name": "child1",
                "parent": None,
                "children": [
                    {
                        "name": "grandchild1",
                        "parent": None,
                        "children": None
                    }
                ]
            },
            {
                "name": "child2",
                "parent": None,
                "children": None
            }
        ]
    }
    
    # Test that the DTO can process the data
    result = backend.transfer_data_from_builtins(tree_data)
    
    assert isinstance(result, TreeNode)
    assert result.name == "root"
    assert result.parent is None
    assert len(result.children) == 2
    assert result.children[0].name == "child1"
    assert result.children[1].name == "child2"
    assert len(result.children[0].children) == 1
    assert result.children[0].children[0].name == "grandchild1"


def test_different_self_referential_models():
    """Test different types of self-referential models."""
    
    # Test linked list
    class LinkedListDTO(DataclassDTO[LinkedListNode]):
        config = DTOConfig(
            experimental_codegen_backend=True,
            max_nested_depth=5,
        )
    
    LinkedListDTO.create_for_field_definition(
        field_definition=FieldDefinition.from_annotation(LinkedListNode),
        handler_id="test-linked-list",
    )
    
    # Get the backend from the DTO class
    backend = LinkedListDTO._dto_backends["test-linked-list"]["return_backend"]
    
    # Create a linked list
    list_data = {
        "value": 1,
        "next": {
            "value": 2,
            "next": {
                "value": 3,
                "next": None
            }
        }
    }
    
    result = backend.transfer_data_from_builtins(list_data)
    
    assert isinstance(result, LinkedListNode)
    assert result.value == 1
    assert result.next.value == 2
    assert result.next.next.value == 3
    assert result.next.next.next is None


if __name__ == "__main__":
    test_self_referential_dto_performance()
    test_self_referential_dto_correctness()
    test_different_self_referential_models()
    print("All tests passed!")