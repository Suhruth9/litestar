"""DTO backends do the heavy lifting of decoding and validating raw bytes into domain models, and
back again, to bytes.
"""

from __future__ import annotations

import linecache
import re
import secrets
import textwrap
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    cast,
)

from msgspec import UNSET

from litestar.dto._backend import DTOBackend
from litestar.dto._types import (
    CollectionType,
    CompositeType,
    MappingType,
    SimpleType,
    TransferDTOFieldDefinition,
    TransferType,
    UnionType,
)
from litestar.utils.helpers import unique_name_for_scope

if TYPE_CHECKING:
    from litestar.connection import ASGIConnection
    from litestar.dto import AbstractDTO
    from litestar.types.serialization import LitestarEncodableType
    from litestar.typing import FieldDefinition

__all__ = ("DTOCodegenBackend",)


class DTOCodegenBackend(DTOBackend):
    __slots__ = (
        "_encode_data",
        "_transfer_to_dict",
        "_transfer_to_model_type",
    )

    def __init__(
        self,
        dto_factory: type[AbstractDTO],
        field_definition: FieldDefinition,
        handler_id: str,
        is_data_field: bool,
        model_type: type[Any],
        wrapper_attribute_name: str | None,
    ) -> None:
        """Create dto backend instance.

        Args:
            dto_factory: The DTO factory class calling this backend.
            field_definition: Parsed type.
            handler_id: The name of the handler that this backend is for.
            is_data_field: Whether the field is a subclass of DTOData.
            model_type: Model type.
            wrapper_attribute_name: If the data that DTO should operate upon is wrapped in a generic datastructure,
              this is the name of the attribute that the data is stored in.
        """
        super().__init__(
            dto_factory=dto_factory,
            field_definition=field_definition,
            handler_id=handler_id,
            is_data_field=is_data_field,
            model_type=model_type,
            wrapper_attribute_name=wrapper_attribute_name,
        )
        self._transfer_to_dict = self._create_transfer_data_fn(
            destination_type=dict,
            field_definition=self.field_definition,
        )
        self._transfer_to_model_type = self._create_transfer_data_fn(
            destination_type=self.model_type,
            field_definition=self.field_definition,
        )
        self._encode_data = self._create_transfer_data_fn(
            destination_type=self.transfer_model_type,
            field_definition=self.field_definition,
        )

    def populate_data_from_builtins(self, builtins: Any, asgi_connection: ASGIConnection) -> Any:
        """Populate model instance from builtin types.

        Args:
            builtins: Builtin type.
            asgi_connection: The current ASGI Connection

        Returns:
            Instance or collection of ``model_type`` instances.
        """
        if self.dto_data_type:
            return self.dto_data_type(
                backend=self,
                data_as_builtins=self._transfer_to_dict(self.parse_builtins(builtins, asgi_connection)),
            )
        return self.transfer_data_from_builtins(self.parse_builtins(builtins, asgi_connection))

    def transfer_data_from_builtins(self, builtins: Any) -> Any:
        """Populate model instance from builtin types.

        Args:
            builtins: Builtin type.

        Returns:
            Instance or collection of ``model_type`` instances.
        """
        return self._transfer_to_model_type(builtins)

    def populate_data_from_raw(self, raw: bytes, asgi_connection: ASGIConnection) -> Any:
        """Parse raw bytes into instance of `model_type`.

        Args:
            raw: bytes
            asgi_connection: The current ASGI Connection

        Returns:
            Instance or collection of ``model_type`` instances.
        """
        if self.dto_data_type:
            return self.dto_data_type(
                backend=self,
                data_as_builtins=self._transfer_to_dict(self.parse_raw(raw, asgi_connection)),
            )
        return self._transfer_to_model_type(self.parse_raw(raw, asgi_connection))

    def encode_data(self, data: Any) -> LitestarEncodableType:
        """Encode data into a ``LitestarEncodableType``.

        Args:
            data: Data to encode.

        Returns:
            Encoded data.
        """
        if self.wrapper_attribute_name:
            wrapped_transfer = self._encode_data(getattr(data, self.wrapper_attribute_name))
            setattr(data, self.wrapper_attribute_name, wrapped_transfer)
            return cast("LitestarEncodableType", data)

        return cast("LitestarEncodableType", self._encode_data(data))

    def _create_transfer_data_fn(
        self,
        destination_type: type[Any],
        field_definition: FieldDefinition,
    ) -> Any:
        """Create instance or iterable of instances of ``destination_type``.

        Args:
            destination_type: the model type received by the DTO on type narrowing.
            field_definition: the parsed type that represents the handler annotation for which the DTO is being applied.

        Returns:
            Data parsed into ``destination_type``.
        """

        return TransferFunctionFactory.create_transfer_data(
            destination_type=destination_type,
            field_definitions=self.parsed_field_definitions,
            is_data_field=self.is_data_field,
            field_definition=field_definition,
            attribute_accessor=self.attribute_accessor,
        )


class FieldAccessManager(Protocol):
    def __call__(self, source_name: str, field_name: str, expect_optional: bool) -> AbstractContextManager[str]: ...


class TransferFunctionFactory:
    # Class-level cache for transfer functions to avoid regenerating for recursive types
    _transfer_function_cache: dict[tuple[Any, ...], Callable[[Any], Any]] = {}
    
    def __init__(
        self,
        is_data_field: bool,
        nested_as_dict: bool,
        attribute_accessor: Callable[[object, str], Any],
        processing_types: set[type[Any]] | None = None,
    ) -> None:
        self.attribute_accessor = attribute_accessor
        self.is_data_field = is_data_field
        self._fn_locals: dict[str, Any] = {
            "Mapping": Mapping,
            "UNSET": UNSET,
        }
        if attribute_accessor is not getattr:
            self.attribute_accessor_name: str | None = self._add_to_fn_globals("__getattr_impl", attribute_accessor)
        else:
            self.attribute_accessor_name = None
        self._indentation = 1
        self._body = ""
        self.names: set[str] = set()
        self.nested_as_dict = nested_as_dict
        self._re_index_access = re.compile(r"\[['\"](\w+?)['\"]]")
        # Track types being processed to detect recursion
        self.processing_types = processing_types or set()

    def _add_to_fn_globals(self, name: str, value: Any) -> str:
        unique_name = unique_name_for_scope(name, self._fn_locals)
        self._fn_locals[unique_name] = value
        return unique_name

    def _create_local_name(self, name: str) -> str:
        unique_name = unique_name_for_scope(name, self.names)
        self.names.add(unique_name)
        return unique_name

    def _make_function(
        self,
        source_value_name: str,
        return_value_name: str,
        fn_name: str = "func",
    ) -> Callable[[Any], Any]:
        """Wrap the current body contents in a function definition and turn it into a callable object"""
        source = f"def {fn_name}({source_value_name}):\n{self._body} return {return_value_name}"
        ctx: dict[str, Any] = {**self._fn_locals}

        # add the function to linecache, to get better stacktraces when an error occurs
        # otherwise, the traceback within the generated code will just point
        # to '<string>'
        file_name = f"dto_transfer_function_{secrets.token_hex(6)}"
        linecache.cache[file_name] = (
            len(source),
            None,  # mtime: not applicable
            [line + "\n" for line in source.splitlines()],
            file_name,
        )
        code = compile(source, file_name, "exec")
        exec(code, ctx)  # noqa: S102

        return ctx["func"]  # type: ignore[no-any-return]

    def _add_stmt(self, stmt: str) -> None:
        self._body += textwrap.indent(stmt + "\n", " " * self._indentation)

    @contextmanager
    def _start_block(self, expr: str | None = None) -> Generator[None, None, None]:
        """Start an indented block. If `expr` is given, use it as the "opening line"
        of the block.
        """
        if expr is not None:
            self._add_stmt(expr)
        self._indentation += 1
        yield
        self._indentation -= 1

    @contextmanager
    def _try_except_pass(self, exception: str) -> Generator[None, None, None]:
        """Enter a `try / except / pass` block. Content written while inside this context
        will go into the `try` block.
        """
        with self._start_block("try:"):
            yield
        with self._start_block(expr=f"except {exception}:"):
            self._add_stmt("pass")

    @contextmanager
    def _access_mapping_item(
        self, source_name: str, field_name: str, expect_optional: bool
    ) -> Generator[str, None, None]:
        """Enter a context within which an item of a mapping can be accessed safely,
        i.e. only if it is contained within that mapping.
        Yields an expression that accesses the mapping item. Content written while
        within this context can use this expression to access the desired value.
        """
        value_expr = f"{source_name}['{field_name}']"

        # if we expect an optional item, it's faster to check if it exists beforehand
        if expect_optional:
            with self._start_block(f"if '{field_name}' in {source_name}:"):
                yield value_expr
        # the happy path of a try/except will be faster than that, so we use that if
        # we expect a value
        else:
            with self._try_except_pass("KeyError"):
                yield value_expr

    @contextmanager
    def _access_attribute(self, source_name: str, field_name: str, expect_optional: bool) -> Generator[str, None, None]:
        """Enter a context within which an attribute of an object can be accessed
        safely, i.e. only if the object actually has the attribute.
        Yields an expression that retrieves the object attribute. Content written while
        within this context can use this expression to access the desired value.
        """

        if self.attribute_accessor_name:
            value_expr = f"{self.attribute_accessor_name}({source_name}, '{field_name}')"
        else:
            value_expr = f"{source_name}.{field_name}"

        # if we expect an optional attribute it's faster to check with hasattr
        if expect_optional:
            with self._start_block(f"if hasattr({source_name}, '{field_name}'):"):
                yield value_expr
        # the happy path of a try/except will be faster than that, so we use that if
        # we expect a value
        else:
            with self._try_except_pass("AttributeError"):
                yield value_expr

    @classmethod
    def create_transfer_instance_data(
        cls,
        field_definitions: tuple[TransferDTOFieldDefinition, ...],
        destination_type: type[Any],
        is_data_field: bool,
        attribute_accessor: Callable[[object, str], Any],
        processing_types: set[type[Any]] | None = None,
    ) -> Callable[[Any], Any]:
        factory = cls(
            is_data_field=is_data_field,
            nested_as_dict=destination_type is dict,
            attribute_accessor=attribute_accessor,
            processing_types=processing_types,
        )
        tmp_return_type_name = factory._create_local_name("tmp_return_type")
        source_instance_name = factory._create_local_name("source_instance")
        destination_type_name = factory._add_to_fn_globals("destination_type", destination_type)
        factory._create_transfer_instance_data(
            tmp_return_type_name=tmp_return_type_name,
            source_instance_name=source_instance_name,
            destination_type_name=destination_type_name,
            field_definitions=field_definitions,
            destination_type_is_dict=destination_type is dict,
        )
        return factory._make_function(source_value_name=source_instance_name, return_value_name=tmp_return_type_name)

    @classmethod
    def create_transfer_type_data(
        cls,
        transfer_type: TransferType,
        is_data_field: bool,
        attribute_accessor: Callable[[object, str], Any],
        processing_types: set[type[Any]] | None = None,
    ) -> Callable[[Any], Any]:
        factory = cls(
            is_data_field=is_data_field,
            nested_as_dict=False,
            attribute_accessor=attribute_accessor,
            processing_types=processing_types,
        )
        tmp_return_type_name = factory._create_local_name("tmp_return_type")
        source_value_name = factory._create_local_name("source_value")
        factory._create_transfer_type_data_body(
            transfer_type=transfer_type,
            nested_as_dict=False,
            assignment_target=tmp_return_type_name,
            source_value_name=source_value_name,
        )
        return factory._make_function(source_value_name=source_value_name, return_value_name=tmp_return_type_name)

    @classmethod
    def create_transfer_data(
        cls,
        *,
        destination_type: type[Any],
        field_definitions: tuple[TransferDTOFieldDefinition, ...],
        is_data_field: bool,
        field_definition: FieldDefinition | None = None,
        attribute_accessor: Callable[[object, str], Any],
        processing_types: set[type[Any]] | None = None,
    ) -> Callable[[Any], Any]:
        processing_types = processing_types or set()
        
        # Check if we're processing a self-referential type
        # Use the destination_type as the key to detect recursion
        if destination_type in processing_types:
            # For self-referential types, create a placeholder that will be replaced
            # with the actual recursive function
            return lambda x: x  # Placeholder that will be replaced
        
        # Create a cache key for this specific transfer function
        cache_key = (
            destination_type,
            tuple((fd.name, fd.transfer_type.__class__.__name__) for fd in field_definitions),
            is_data_field,
            field_definition.annotation if field_definition else None,
        )
        
        # Check if we've already created this transfer function
        if cache_key in cls._transfer_function_cache:
            return cls._transfer_function_cache[cache_key]
        
        # Add current type to processing set
        new_processing_types = processing_types | {destination_type}
        
        if field_definition and field_definition.is_non_string_collection:
            factory = cls(
                is_data_field=is_data_field,
                nested_as_dict=False,
                attribute_accessor=attribute_accessor,
                processing_types=new_processing_types,
            )
            source_value_name = factory._create_local_name("source_value")
            return_value_name = factory._create_local_name("tmp_return_value")
            factory._create_transfer_data_body_nested(
                field_definitions=field_definitions,
                field_definition=field_definition,
                destination_type=destination_type,
                source_data_name=source_value_name,
                assignment_target=return_value_name,
            )
            result = factory._make_function(source_value_name=source_value_name, return_value_name=return_value_name)
            cls._transfer_function_cache[cache_key] = result
            return result

        result = cls.create_transfer_instance_data(
            destination_type=destination_type,
            field_definitions=field_definitions,
            is_data_field=is_data_field,
            attribute_accessor=attribute_accessor,
            processing_types=new_processing_types,
        )
        cls._transfer_function_cache[cache_key] = result
        return result

    def _create_transfer_data_body_nested(
        self,
        field_definition: FieldDefinition,
        field_definitions: tuple[TransferDTOFieldDefinition, ...],
        destination_type: type[Any],
        source_data_name: str,
        assignment_target: str,
    ) -> None:
        origin_name = self._add_to_fn_globals("origin", field_definition.instantiable_origin)
        
        # Check if this is a self-referential type
        # We need to check if the inner type is the same as the destination type
        # AND that we're already processing this type (to detect recursion)
        inner_field = field_definition.inner_types[0] if field_definition.inner_types else None
        is_self_referential = (
            inner_field and 
            inner_field.annotation == destination_type and
            destination_type in self.processing_types
        )
        
        if is_self_referential:
            # For self-referential types, use the same function recursively
            # This avoids generating new functions for each depth level
            transfer_func_name = "func"  # Reference to the function being generated
            if field_definition.is_mapping:
                self._add_stmt(
                    f"{assignment_target} = {origin_name}((key, {transfer_func_name}(item)) for key, item in {source_data_name}.items())"
                )
            else:
                self._add_stmt(
                    f"{assignment_target} = {origin_name}({transfer_func_name}(item) for item in {source_data_name})"
                )
        else:
            # For non-self-referential types, create a separate transfer function
            transfer_func = TransferFunctionFactory.create_transfer_data(
                is_data_field=self.is_data_field,
                destination_type=destination_type,
                field_definition=field_definition.inner_types[0],
                field_definitions=field_definitions,
                attribute_accessor=self.attribute_accessor,
                processing_types=self.processing_types,
            )
            transfer_func_name = self._add_to_fn_globals("transfer_data", transfer_func)
            if field_definition.is_mapping:
                self._add_stmt(
                    f"{assignment_target} = {origin_name}((key, {transfer_func_name}(item)) for key, item in {source_data_name}.items())"
                )
            else:
                self._add_stmt(
                    f"{assignment_target} = {origin_name}({transfer_func_name}(item) for item in {source_data_name})"
                )

    def _create_transfer_instance_data(
        self,
        tmp_return_type_name: str,
        source_instance_name: str,
        destination_type_name: str,
        field_definitions: tuple[TransferDTOFieldDefinition, ...],
        destination_type_is_dict: bool,
    ) -> None:
        local_dict_name = self._create_local_name("unstructured_data")
        self._add_stmt(f"{local_dict_name} = {{}}")

        if field_definitions := tuple(f for f in field_definitions if self.is_data_field or not f.is_excluded):
            if len(field_definitions) > 1 and ("." in source_instance_name or "[" in source_instance_name):
                # If there's more than one field we have to access, we check if it is
                # nested. If it is nested, we assign it to a local variable to avoid
                # repeated lookups. This is only a small performance improvement for
                # regular attributes, but can be quite significant for properties or
                # other types of descriptors, where I/O may be involved, such as the
                # case for lazy loaded relationships in SQLAlchemy
                if "." in source_instance_name:
                    level_1, level_2 = source_instance_name.split(".", 1)
                else:
                    level_1, level_2, *_ = self._re_index_access.split(source_instance_name, maxsplit=1)

                new_source_instance_name = self._create_local_name(f"{level_1}_{level_2}")
                self._add_stmt(f"{new_source_instance_name} = {source_instance_name}")
                source_instance_name = new_source_instance_name

            for source_type in ("mapping", "object"):
                if source_type == "mapping":
                    block_expr = f"if isinstance({source_instance_name}, Mapping):"
                    access_item = self._access_mapping_item
                else:
                    block_expr = "else:"
                    access_item = self._access_attribute

                with self._start_block(expr=block_expr):
                    self._create_transfer_instance_data_inner(
                        local_dict_name=local_dict_name,
                        field_definitions=field_definitions,
                        access_field_safe=access_item,
                        source_instance_name=source_instance_name,
                    )

        # if the destination type is a dict we can reuse our temporary dictionary of
        # unstructured data as the "return value"
        if not destination_type_is_dict:
            self._add_stmt(f"{tmp_return_type_name} = {destination_type_name}(**{local_dict_name})")
        else:
            self._add_stmt(f"{tmp_return_type_name} = {local_dict_name}")

    def _create_transfer_instance_data_inner(
        self,
        *,
        local_dict_name: str,
        field_definitions: tuple[TransferDTOFieldDefinition, ...],
        access_field_safe: FieldAccessManager,
        source_instance_name: str,
    ) -> None:
        for field_definition in field_definitions:
            with access_field_safe(
                source_name=source_instance_name,
                field_name=field_definition.name,
                expect_optional=field_definition.is_partial or field_definition.is_optional,
            ) as source_value_expr:
                if self.is_data_field and field_definition.is_partial:
                    # we assign the source value to a name here, so we can skip
                    # getting it twice from the source instance
                    source_value_name = self._create_local_name("source_value")
                    self._add_stmt(f"{source_value_name} = {source_value_expr}")
                    ctx = self._start_block(f"if {source_value_name} is not UNSET:")
                else:
                    # in these cases, we only ever access the source value once, so
                    # we can skip assigning it
                    source_value_name = source_value_expr
                    ctx = nullcontext()  # type: ignore[assignment]
                with ctx:
                    self._create_transfer_type_data_body(
                        transfer_type=field_definition.transfer_type,
                        nested_as_dict=self.nested_as_dict,
                        source_value_name=source_value_name,
                        assignment_target=f"{local_dict_name}['{field_definition.name}']",
                    )

    def _create_transfer_type_data_body(
        self,
        transfer_type: TransferType,
        nested_as_dict: bool,
        source_value_name: str,
        assignment_target: str,
    ) -> None:
        if isinstance(transfer_type, SimpleType) and transfer_type.nested_field_info:
            if nested_as_dict:
                destination_type: Any = dict
            elif self.is_data_field:
                destination_type = transfer_type.field_definition.annotation
            else:
                destination_type = transfer_type.nested_field_info.model

            self._create_transfer_instance_data(
                field_definitions=transfer_type.nested_field_info.field_definitions,
                tmp_return_type_name=assignment_target,
                source_instance_name=source_value_name,
                destination_type_name=self._add_to_fn_globals("destination_type", destination_type),
                destination_type_is_dict=destination_type is dict,
            )
            return

        if isinstance(transfer_type, UnionType) and transfer_type.has_nested:
            self._create_transfer_nested_union_type_data(
                transfer_type=transfer_type,
                source_value_name=source_value_name,
                assignment_target=assignment_target,
            )
            return

        if isinstance(transfer_type, CollectionType):
            origin_name = self._add_to_fn_globals("origin", transfer_type.field_definition.instantiable_origin)
            if transfer_type.has_nested:
                transfer_type_data_fn = TransferFunctionFactory.create_transfer_type_data(
                    is_data_field=self.is_data_field,
                    transfer_type=transfer_type.inner_type,
                    attribute_accessor=self.attribute_accessor,
                    processing_types=self.processing_types,
                )
                transfer_type_data_name = self._add_to_fn_globals("transfer_type_data", transfer_type_data_fn)
                self._add_stmt(
                    f"{assignment_target} = {origin_name}({transfer_type_data_name}(item) for item in {source_value_name})"
                )
                return

            self._add_stmt(f"{assignment_target} = {origin_name}({source_value_name})")
            return

        if isinstance(transfer_type, MappingType):
            origin_name = self._add_to_fn_globals("origin", transfer_type.field_definition.instantiable_origin)
            if transfer_type.has_nested:
                transfer_type_data_fn = TransferFunctionFactory.create_transfer_type_data(
                    is_data_field=self.is_data_field,
                    transfer_type=transfer_type.value_type,
                    attribute_accessor=self.attribute_accessor,
                    processing_types=self.processing_types,
                )
                transfer_type_data_name = self._add_to_fn_globals("transfer_type_data", transfer_type_data_fn)
                self._add_stmt(
                    f"{assignment_target} = {origin_name}((key, {transfer_type_data_name}(item)) for key, item in {source_value_name}.items())"
                )
                return

            self._add_stmt(f"{assignment_target} = {origin_name}({source_value_name})")
            return

        self._add_stmt(f"{assignment_target} = {source_value_name}")

    def _create_transfer_nested_union_type_data(
        self,
        transfer_type: UnionType,
        source_value_name: str,
        assignment_target: str,
    ) -> None:
        for inner_type in transfer_type.inner_types:
            if isinstance(inner_type, CompositeType):
                continue

            if inner_type.nested_field_info:
                if self.is_data_field:
                    constraint_type = inner_type.nested_field_info.model
                    destination_type = inner_type.field_definition.annotation
                else:
                    constraint_type = inner_type.field_definition.annotation
                    destination_type = inner_type.nested_field_info.model

                constraint_type_name = self._add_to_fn_globals("constraint_type", constraint_type)
                destination_type_name = self._add_to_fn_globals("destination_type", destination_type)

                with self._start_block(f"if isinstance({source_value_name}, {constraint_type_name}):"):
                    self._create_transfer_instance_data(
                        destination_type_name=destination_type_name,
                        destination_type_is_dict=destination_type is dict,
                        field_definitions=inner_type.nested_field_info.field_definitions,
                        source_instance_name=source_value_name,
                        tmp_return_type_name=assignment_target,
                    )
                    return
        self._add_stmt(f"{assignment_target} = {source_value_name}")
