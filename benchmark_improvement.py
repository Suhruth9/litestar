"""Benchmark the performance improvement for self-referential DTOs."""
import time
from dataclasses import dataclass

from litestar.dto import DataclassDTO, DTOConfig
from litestar.typing import FieldDefinition


@dataclass
class LeafNode:
    name: str
    parent: "LeafNode | None" = None
    children: "list[LeafNode] | None" = None


def benchmark_depth(depth: int, use_codegen: bool = True) -> float:
    """Benchmark DTO creation for a specific depth."""
    class TestDTO(DataclassDTO[LeafNode]):
        config = DTOConfig(
            experimental_codegen_backend=use_codegen,
            max_nested_depth=depth,
        )
    
    start = time.perf_counter()
    TestDTO.create_for_field_definition(
        field_definition=FieldDefinition.from_annotation(LeafNode),
        handler_id=f"benchmark-{depth}-{use_codegen}",
    )
    return time.perf_counter() - start


def main():
    print("Performance Benchmark for Self-Referential DTO Optimization")
    print("=" * 60)
    
    depths = [3, 5, 7, 9]
    
    print("\nWith Codegen Backend (Optimized):")
    print("-" * 40)
    codegen_times = []
    for depth in depths:
        elapsed = benchmark_depth(depth, use_codegen=True)
        codegen_times.append(elapsed)
        print(f"  Depth {depth:2d}: {elapsed:8.4f} seconds")
        # Clear cache between tests
        from litestar.dto._codegen_backend import TransferFunctionFactory
        TransferFunctionFactory._transfer_function_cache.clear()
    
    print("\nWithout Codegen Backend:")
    print("-" * 40)
    no_codegen_times = []
    for depth in depths:
        elapsed = benchmark_depth(depth, use_codegen=False)
        no_codegen_times.append(elapsed)
        print(f"  Depth {depth:2d}: {elapsed:8.4f} seconds")
    
    print("\nPerformance Analysis:")
    print("-" * 40)
    
    # Calculate growth factors
    if len(codegen_times) > 1:
        for i in range(1, len(codegen_times)):
            growth = codegen_times[i] / codegen_times[i-1] if codegen_times[i-1] > 0 else 0
            print(f"  Codegen growth from depth {depths[i-1]} to {depths[i]}: {growth:.2f}x")
    
    print()
    
    # Compare codegen vs no-codegen
    print("Codegen vs No-Codegen comparison:")
    for i, depth in enumerate(depths):
        if no_codegen_times[i] > 0:
            ratio = codegen_times[i] / no_codegen_times[i]
            if ratio < 1:
                print(f"  Depth {depth}: Codegen is {1/ratio:.2f}x faster")
            else:
                print(f"  Depth {depth}: Codegen is {ratio:.2f}x slower")
    
    # Check if optimization is working (growth should be sub-exponential)
    print("\nOptimization Status:")
    print("-" * 40)
    
    if len(codegen_times) >= 3:
        # Calculate average growth factor
        growth_factors = []
        for i in range(1, len(codegen_times)):
            if codegen_times[i-1] > 0:
                growth_factors.append(codegen_times[i] / codegen_times[i-1])
        
        avg_growth = sum(growth_factors) / len(growth_factors) if growth_factors else 0
        
        if avg_growth < 4.0:
            print(f"✓ Optimization is working! Average growth factor: {avg_growth:.2f}")
            print("  (Growth factor < 4.0 indicates sub-exponential growth)")
        else:
            print(f"✗ Optimization may need improvement. Average growth factor: {avg_growth:.2f}")
            print("  (Growth factor >= 4.0 indicates possible exponential growth)")
    
    print("\nConclusion:")
    print("-" * 40)
    
    # Determine best approach for different depths
    best_approach = {}
    for i, depth in enumerate(depths):
        if codegen_times[i] < no_codegen_times[i]:
            best_approach[depth] = "codegen"
        else:
            best_approach[depth] = "no-codegen"
    
    print("Recommended approach by depth:")
    for depth in depths:
        print(f"  Depth {depth}: Use {best_approach[depth]}")


if __name__ == "__main__":
    main()