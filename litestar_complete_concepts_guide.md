# Complete Litestar Concepts Learning Guide

Let's learn all 100 concepts systematically, with Litestar-specific examples for each. I'll organize them in a logical learning order where each concept builds on previous ones.

## Part 1: Python Type System Fundamentals

### 1. **Type Hints** - The Foundation
Type hints tell Python (and developers) what type of data a function expects and returns.

```python
# Basic Python example
def greet(name: str) -> str:
    return f"Hello, {name}"

# Litestar example from handlers
from litestar import get

@get("/users/{user_id:int}")
async def get_user(user_id: int) -> dict[str, Any]:
    # Litestar uses type hints to:
    # 1. Validate user_id is an integer
    # 2. Convert string from URL to int
    # 3. Serialize return dict to JSON
    return {"id": user_id, "name": "John"}
```

**Why it matters in Litestar**: Type hints drive EVERYTHING - validation, serialization, dependency injection, and OpenAPI documentation.

### 2. **Optional Types**
Indicates a value can be the specified type OR None.

```python
# Two ways to write optional
from typing import Optional

# Old style
name: Optional[str] = None

# New style (Python 3.10+)
name: str | None = None

# Litestar example
@get("/search")
async def search(
    query: str,                    # Required
    limit: int = 10,               # Optional with default
    category: str | None = None    # Optional, can be None
) -> list[dict]:
    # Litestar automatically makes category optional in API
    if category:
        return search_with_category(query, category, limit)
    return search_all(query, limit)
```

### 3. **Union Types**
A value can be one of several types.

```python
# Python basics
from typing import Union

# Old style
result: Union[int, str, None]

# New style  
result: int | str | None

# Litestar usage in DTOs
from litestar.dto import DTOFieldDefinition

# DTOs check for unions
if field_definition.is_union and not field_definition.is_optional:
    raise InvalidAnnotationException("Unions not supported")

# Why? Complex unions make serialization ambiguous
# But Optional (Union[T, None]) is fine
```

### 4. **Generic Types**
Types that can work with any type parameter.

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

# Litestar's DTO system
from litestar.dto import AbstractDTO

class AbstractDTO(Generic[T]):
    model_type: type[T]
    
    def decode_builtins(self, value: dict[str, Any]) -> T:
        # Returns instance of type T
        pass

# Usage
UserDTO = AbstractDTO[User]  # Specialized for User model
PostDTO = AbstractDTO[Post]  # Specialized for Post model
```

### 5. **TypeVar**
Defines a type variable for generics.

```python
from typing import TypeVar, Callable

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)      # Can be subclass
T_contra = TypeVar('T_contra', contravariant=True)  # Can be superclass

# Litestar example
def ensure_list(value: T | list[T]) -> list[T]:
    """Convert single value or list to list."""
    if isinstance(value, list):
        return value
    return [value]

# Constrained TypeVar
HandlerType = TypeVar(
    'HandlerType',
    bound=BaseRouteHandler  # Must be subclass of BaseRouteHandler
)
```

### 6. **ClassVar**
Indicates a variable belongs to the class, not instances.

```python
from typing import ClassVar

class DTOBackend:
    # Shared across all instances
    _seen_model_names: ClassVar[set[str]] = set()
    
    # Instance variable
    handler_id: str
    
    def __init__(self, handler_id: str):
        self.handler_id = handler_id
        # Can access class variable
        if handler_id in self._seen_model_names:
            print("Already seen!")
        self._seen_model_names.add(handler_id)
```

### 7. **Final**
Indicates a value shouldn't be reassigned.

```python
from typing import Final

class DTOBackend:
    # These are set once in __init__ and never changed
    dto_factory: Final[type[AbstractDTO]]
    field_definition: Final[FieldDefinition]
    handler_id: Final[str]
    
    def __init__(self, dto_factory, field_definition, handler_id):
        self.dto_factory = dto_factory  # Set once
        self.field_definition = field_definition
        self.handler_id = handler_id
        # These should never be reassigned after this
```

### 8. **TypedDict**
Dictionary with specific keys and value types.

```python
from typing import TypedDict, NotRequired

class HandlerIndex(TypedDict):
    """Used in Litestar for route handler mapping."""
    paths: list[str]
    handler: RouteHandlerType
    identifier: str

# With optional keys
class _BackendDict(TypedDict):
    data_backend: NotRequired[DTOBackend]
    return_backend: NotRequired[DTOBackend]

# Usage
index: HandlerIndex = {
    "paths": ["/users", "/users/{id}"],
    "handler": get_user_handler,
    "identifier": "get_user"
}
```

### 9. **Protocol Classes**
Define structural subtyping (duck typing with types).

```python
from typing import Protocol

class PluginProtocol(Protocol):
    """Any class with these methods can be a plugin."""
    
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        ...

class InitPluginProtocol(Protocol):
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        ...

# Any class implementing the method works
class MyPlugin:  # Note: doesn't inherit from Protocol
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        return app_config

# Litestar checks protocol conformance
plugin: PluginProtocol = MyPlugin()  # Works!
```

### 10. **TYPE_CHECKING**
Import guard for type checking only.

```python
from typing import TYPE_CHECKING

# These imports only happen during type checking
# Prevents circular imports at runtime
if TYPE_CHECKING:
    from litestar.connection import Request
    from litestar.response import Response

def handler(request: "Request") -> "Response":
    # At runtime, Request and Response aren't imported
    # But type checkers see them
    pass
```

### 11. **Annotated**
Adds metadata to type hints.

```python
from typing import Annotated
from litestar.params import Parameter

# Basic annotation
UserId = Annotated[int, "User ID in database"]

# Litestar uses for constraints
from msgspec import Meta

def create_field_type(field_def):
    field_type = int
    
    # Add validation metadata
    field_meta = Meta(
        gt=0,  # Greater than 0
        le=1000,  # Less than or equal to 1000
    )
    
    return Annotated[field_type, field_meta]

# In handlers
@get("/users/{user_id:int}")
async def get_user(
    user_id: Annotated[int, Parameter(gt=0, le=1000)]
) -> User:
    pass
```

### 12. **Forward References**
String type hints for not-yet-defined types.

```python
# Problem: class references itself
class Node:
    def __init__(self, value: int, next: Node):  # Error! Node not defined yet
        pass

# Solution: string forward reference
class Node:
    def __init__(self, value: int, next: "Node"):
        pass

# Litestar checks for these
if field_definition.is_forward_ref:
    raise InvalidAnnotationException(
        "Forward references are not supported as type argument to DTO"
    )
# Why? Can't introspect string types at runtime
```

## Part 2: Object-Oriented Advanced

### 13. **`__slots__`**
Restricts attributes and saves memory.

```python
# Without slots - uses __dict__
class RegularClass:
    def __init__(self):
        self.x = 1
        self.y = 2
        self.dynamic = 3  # Can add any attribute

# With slots - no __dict__
class Litestar(Router):
    __slots__ = (
        "_debug",
        "_openapi_schema",
        "asgi_handler",
        "asgi_router",
        # ... all attributes listed
    )
    
    def __init__(self):
        self._debug = False
        # self.new_attr = 1  # ERROR! Not in __slots__

# Benefits:
# 1. 40% less memory usage
# 2. Faster attribute access
# 3. Prevents typos (can't accidentally create new attributes)
```

### 14. **`__class_getitem__`**
Makes classes subscriptable for generics.

```python
class AbstractDTO:
    def __class_getitem__(cls, annotation: Any) -> type[Self]:
        """Called when you do AbstractDTO[User]"""
        
        # Parse the annotation
        field_definition = FieldDefinition.from_annotation(annotation)
        
        # Create specialized subclass
        cls_dict = {
            "model_type": annotation,
            "config": DTOConfig()
        }
        
        # Return new class specialized for this type
        return type(
            f"{cls.__name__}[{annotation.__name__}]",
            (cls,),  # Inherit from cls
            cls_dict  # With these attributes
        )

# Usage
UserDTO = AbstractDTO[User]  # Creates specialized class
PostDTO = AbstractDTO[Post]  # Different specialized class
```

### 15. **`__init_subclass__`**
Hook called when class is subclassed.

```python
class AbstractDTO:
    def __init_subclass__(cls, **kwargs):
        """Called when someone creates: class MyDTO(AbstractDTO)"""
        
        # Configure the subclass
        if config := getattr(cls, "config", None):
            if model_type := getattr(cls, "model_type", None):
                # Set up the concrete class
                cls.config = cls.get_config_for_model_type(
                    config, model_type
                )

# When you subclass:
class UserDTO(AbstractDTO):
    config = DTOConfig(exclude=["password"])
    model_type = User
    # __init_subclass__ automatically configures this
```

### 16. **Abstract Base Classes (ABC)**
Define interfaces that must be implemented.

```python
from abc import ABC, abstractmethod

class AbstractDTO(ABC):
    @abstractmethod
    def generate_field_definitions(
        self, model_type: type[Any]
    ) -> Generator[DTOFieldDefinition, None, None]:
        """Subclasses MUST implement this."""
        pass
    
    @abstractmethod
    def detect_nested_field(
        self, field_definition: FieldDefinition
    ) -> bool:
        """Subclasses MUST implement this."""
        pass

# Concrete implementation
class DataclassDTO(AbstractDTO):
    def generate_field_definitions(self, model_type):
        # Must provide implementation
        for field in dataclasses.fields(model_type):
            yield DTOFieldDefinition(...)
    
    def detect_nested_field(self, field_definition):
        return dataclasses.is_dataclass(field_definition.annotation)
```

### 17. **Property Decorators and Setters**

```python
class Litestar:
    def __init__(self):
        self._debug = False
    
    @property
    def debug(self) -> bool:
        """Getter - access like attribute."""
        return self._debug
    
    @debug.setter
    def debug(self, value: bool) -> None:
        """Setter - customize assignment behavior."""
        if self.logger and self.logging_config:
            # Update logger when debug changes
            level = logging.DEBUG if value else logging.INFO
            self.logging_config.set_level(self.logger, level)
        self._debug = value

# Usage
app = Litestar()
print(app.debug)  # Calls getter
app.debug = True  # Calls setter, updates logger
```

### 18. **Class Methods vs Static Methods**

```python
class AbstractDTO:
    # Class method - receives class as first argument
    @classmethod
    def get_config_for_model_type(
        cls,  # The class itself
        config: DTOConfig,
        model_type: type[Any]
    ) -> DTOConfig:
        # Can access class attributes
        return config
    
    # Static method - no automatic first argument
    @staticmethod
    def _get_default_plugins(plugins: list) -> list:
        # Just a regular function in class namespace
        # Can't access class or instance
        plugins.append(MsgspecDIPlugin())
        return plugins

# Usage
config = AbstractDTO.get_config_for_model_type(cfg, User)
plugins = AbstractDTO._get_default_plugins([])
```

### 19. **Multiple Inheritance & MRO**

```python
# Litestar uses single inheritance mostly, but here's the concept
class Router:
    def register(self): pass

class EventEmitter:
    def emit(self): pass

class Litestar(Router):  # Single inheritance in Litestar
    # But could have been:
    # class Litestar(Router, EventEmitter):
    pass

# Method Resolution Order (MRO)
print(Litestar.__mro__)
# (Litestar, Router, object)

# Diamond problem handled by Python's C3 linearization
```

## Part 3: Decorators & Functions

### 20. **Function Decorators**

```python
# Basic decorator
def ensure_async(func):
    """Decorator that makes function async."""
    if asyncio.iscoroutinefunction(func):
        return func
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper

# Litestar's route decorators
def get(path: str):
    def decorator(func):
        return HTTPRouteHandler(
            fn=func,
            path=path,
            http_method=HttpMethod.GET
        )
    return decorator

# Usage
@get("/users")
async def get_users() -> list[User]:
    pass
```

### 21. **Decorator Factories**

```python
# Decorator that takes parameters
def route(
    path: str,
    method: str,
    cache: bool = False
):
    def decorator(func):
        handler = HTTPRouteHandler(
            fn=func,
            path=path,
            http_method=method
        )
        if cache:
            handler.enable_cache()
        return handler
    
    return decorator

# Usage
@route("/users", "GET", cache=True)
async def get_users():
    pass
```

### 22. **Class Decorators**

```python
# Decorator that modifies classes
def dataclass_decorator(cls):
    """Add DTO functionality to dataclass."""
    cls.__dto_fields__ = []
    for field in dataclasses.fields(cls):
        cls.__dto_fields__.append(field.name)
    return cls

@dataclass_decorator
@dataclass
class User:
    id: int
    name: str
    # Now has __dto_fields__ attribute
```

### 23. **Descriptor Protocol**

```python
class DTOProperty:
    """Descriptor for DTO fields."""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._data.get(self.field_name)
    
    def __set__(self, obj, value):
        obj._data[self.field_name] = value

class Model:
    name = DTOProperty("name")
    age = DTOProperty("age")
    
    def __init__(self):
        self._data = {}
```

## Part 4: Async Programming

### 24. **async/await Basics**

```python
# Synchronous
def get_user_sync(user_id: int) -> User:
    user = database.query(f"SELECT * FROM users WHERE id={user_id}")
    return user

# Asynchronous
async def get_user_async(user_id: int) -> User:
    user = await database.async_query(f"SELECT * FROM users WHERE id={user_id}")
    return user

# Litestar handles both
@get("/users/{user_id:int}")
async def async_handler(user_id: int) -> User:
    return await get_user_async(user_id)

@get("/users-sync/{user_id:int}")
def sync_handler(user_id: int) -> User:
    # Litestar runs this in thread pool
    return get_user_sync(user_id)
```

### 25. **AsyncContextManager**

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_session():
    """Async context manager for database."""
    session = await create_session()
    try:
        yield session
    finally:
        await session.close()

# Usage in Litestar
async with database_session() as session:
    user = await session.get(User, user_id)

# Class-based
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
```

### 26. **AsyncGenerator**

```python
async def stream_users() -> AsyncGenerator[User, None]:
    """Stream users from database."""
    async with database_session() as session:
        query = "SELECT * FROM users"
        async for row in session.stream(query):
            yield User(**row)

# Litestar streaming response
from litestar.response import Stream

@get("/stream-users")
async def stream_handler() -> Stream:
    return Stream(stream_users())
```

### 27. **AsyncExitStack**

```python
from contextlib import AsyncExitStack

async def lifespan(app: Litestar) -> AsyncGenerator[None, None]:
    """Manage multiple async resources."""
    async with AsyncExitStack() as stack:
        # Add multiple contexts
        db = await stack.enter_async_context(database_connection())
        cache = await stack.enter_async_context(redis_connection())
        
        # Register cleanup callbacks
        stack.push_async_callback(cleanup_temp_files)
        
        # Add shutdown hooks in reverse order
        for hook in reversed(app.on_shutdown):
            stack.push_async_callback(hook)
        
        yield  # App runs here
        
        # Everything cleaned up automatically in reverse order
```

### 28. **ASGI Protocol**

```python
# ASGI application signature
async def app(scope: Scope, receive: Receive, send: Send) -> None:
    """
    scope: Connection information (path, headers, etc.)
    receive: Async callable to receive messages
    send: Async callable to send messages
    """
    
    if scope["type"] == "http":
        # Handle HTTP
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
        
        # Send response
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello, World!",
        })

# Litestar's ASGI implementation
class Litestar:
    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self.lifespan_handler(receive, send)
        else:
            await self.asgi_handler(scope, receive, send)
```

## Part 5: Context Managers

### 29. **contextmanager Decorator**

```python
from contextlib import contextmanager

@contextmanager
def temporary_config(app: Litestar, **config):
    """Temporarily change app config."""
    old_config = {}
    
    # Setup
    for key, value in config.items():
        old_config[key] = getattr(app, key)
        setattr(app, key, value)
    
    try:
        yield app
    finally:
        # Cleanup
        for key, value in old_config.items():
            setattr(app, key, value)

# Usage
with temporary_config(app, debug=True):
    # App is in debug mode here
    test_something()
# Debug mode restored
```

### 30. **ExitStack**

```python
from contextlib import ExitStack

def process_files(filenames: list[str]):
    """Process multiple files safely."""
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(fname))
            for fname in filenames
        ]
        
        # All files open here
        process(files)
        
    # All files closed automatically
```

### 31. **AbstractContextManager**

```python
from contextlib import AbstractContextManager

class DatabaseSession(AbstractContextManager):
    def __enter__(self):
        self.session = create_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
        return False  # Don't suppress exceptions
```

## Part 6: Advanced Functions

### 32. **partial from functools**

```python
from functools import partial

# Create specialized functions
def log(level: str, message: str):
    print(f"[{level}] {message}")

# Create specialized versions
debug = partial(log, "DEBUG")
error = partial(log, "ERROR")

# Litestar usage
async def lifespan(app):
    async with AsyncExitStack() as stack:
        # Create partial for each shutdown hook
        for hook in app.on_shutdown[::-1]:
            stack.push_async_callback(
                partial(app._call_lifespan_hook, hook)
            )
```

### 33. **Keyword-only Arguments**

```python
# Force keyword arguments after *
def route(
    path: str,
    *,  # Everything after this is keyword-only
    method: str,
    cache: bool = False,
    guards: list = None
):
    pass

# Usage
# route("/users", "GET")  # ERROR!
route("/users", method="GET")  # Must use keyword

# Litestar decorators use this pattern
def get(
    path: str | None = None,
    *,  # Force keywords for clarity
    after_request: Callable | None = None,
    cache: bool = False,
    # ... 40+ more parameters
):
    pass
```

### 34. **`*args` and `**kwargs`**

```python
def wrapper(*args, **kwargs):
    """Accept any arguments."""
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
    return original_function(*args, **kwargs)

# Litestar usage
def emit(self, event_id: str, *args: Any, **kwargs: Any) -> None:
    """Emit event with any arguments."""
    for listener in self.listeners.get(event_id, []):
        listener(*args, **kwargs)

# Unpacking
def func(a, b, c):
    pass

args = (1, 2, 3)
func(*args)  # Unpacks to func(1, 2, 3)

kwargs = {"a": 1, "b": 2, "c": 3}
func(**kwargs)  # Unpacks to func(a=1, b=2, c=3)
```

### 35. **Callable Types**

```python
from typing import Callable

# Type for any callable
AnyCallable = Callable[..., Any]

# Specific signature
Handler = Callable[[Request], Response]
AsyncHandler = Callable[[Request], Awaitable[Response]]

# Litestar guards
Guard = Callable[[ASGIConnection, BaseRouteHandler], bool]

# Usage
def requires_auth(connection: ASGIConnection, handler: BaseRouteHandler) -> bool:
    return bool(connection.user)

guards: list[Guard] = [requires_auth]
```

### 36. **Generator Functions**

```python
def generate_field_definitions(
    model_type: type
) -> Generator[DTOFieldDefinition, None, None]:
    """Generate field definitions lazily."""
    for field in dataclasses.fields(model_type):
        # Yield one at a time (lazy evaluation)
        yield DTOFieldDefinition(
            name=field.name,
            type=field.type,
            default=field.default
        )

# Can send values to generator
def interactive_generator():
    value = None
    while True:
        value = yield value
        if value is not None:
            value = value * 2

gen = interactive_generator()
next(gen)  # Prime the generator
gen.send(5)  # Returns 10
```

## Part 7: Data Structures

### 37. **dataclasses**

```python
from dataclasses import dataclass, field

@dataclass
class User:
    id: int
    name: str
    email: str
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

# Litestar DTO integration
from litestar.dto import DataclassDTO

class UserDTO(DataclassDTO[User]):
    config = DTOConfig(exclude={"created_at"})

@post("/users", dto=UserDTO)
async def create_user(data: User) -> User:
    # Automatic validation and serialization
    return data
```

### 38. **NamedTuple**

```python
from typing import NamedTuple

class RouteMapItem(NamedTuple):
    path: str
    methods: set[str]
    handler: Callable

# Immutable and lightweight
item = RouteMapItem("/users", {"GET", "POST"}, handler_func)
path, methods, handler = item  # Unpacking
```

### 39. **Enum Classes**

```python
from enum import Enum

class HttpMethod(str, Enum):
    """HTTP methods as enum."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

# Usage in Litestar
@route(path="/users", http_method=HttpMethod.GET)
async def handler():
    pass

# Can iterate
for method in HttpMethod:
    print(method.value)
```

### 40. **defaultdict**

```python
from collections import defaultdict

# Litestar groups handlers by path
def build_routes(handlers):
    http_path_groups = defaultdict(list)
    
    for handler in handlers:
        for path in handler.paths:
            # No need to check if key exists
            http_path_groups[path].append(handler)
    
    return http_path_groups

# Regular dict would need:
# if path not in groups:
#     groups[path] = []
# groups[path].append(handler)
```

### 41. **ChainMap** (Layered Dictionaries)

```python
from collections import ChainMap

# Layered configuration (not directly used in Litestar but concept applies)
app_config = {"debug": False, "timeout": 30}
router_config = {"timeout": 60}
handler_config = {"timeout": 120, "cache": True}

# Creates layered view
config = ChainMap(handler_config, router_config, app_config)
print(config["timeout"])  # 120 (from handler)
print(config["debug"])    # False (from app)
print(config["cache"])    # True (from handler)
```

## Part 8: Modern Python Features

### 42. **Walrus Operator `:=`**

```python
# Assign and use in same expression
# Old way
parts = text.split(".")
if len(parts) > 1:
    process(parts)

# With walrus
if (parts := text.split(".")) and len(parts) > 1:
    process(parts)

# Litestar example
def _filter_nested_field(field_name_set: Set[str], field_name: str) -> Set[str]:
    return {
        split[1] 
        for s in field_name_set 
        if (split := s.split(".", 1))[0] == field_name and len(split) > 1
    }
```

### 43. **Match/Case (Pattern Matching)**

```python
# Python 3.10+ feature (not in Litestar yet but useful)
def process_route(route):
    match route:
        case HTTPRoute(path=path, methods=methods):
            return f"HTTP route: {path}"
        case WebSocketRoute(path=path):
            return f"WebSocket route: {path}"
        case _:
            return "Unknown route"
```

### 44. **f-strings**

```python
# Formatted string literals
name = "Litestar"
version = "2.0"

# Basic
message = f"Using {name} version {version}"

# Expressions
result = f"Sum: {2 + 2}"

# Format specifiers
pi = 3.14159
formatted = f"Pi: {pi:.2f}"  # "Pi: 3.14"

# Debug (3.8+)
value = 42
debug = f"{value=}"  # "value=42"

# Litestar usage
return type(
    f"{cls.__name__}[{annotation}]",  # Dynamic class name
    (cls,),
    cls_dict
)
```

### 45. **Union Operators for Types**

```python
# Old way
from typing import Union, Optional

result: Union[int, str]
maybe: Optional[int]  # Union[int, None]

# New way (3.10+)
result: int | str
maybe: int | None

# Can use with isinstance
value = "hello"
if isinstance(value, int | str):
    print("Valid type")
```

## Part 9: Software Design Patterns

### 46. **Factory Pattern**

```python
class DTOFactory:
    """Factory for creating DTOs."""
    
    @staticmethod
    def create_dto(model_type: type) -> type[AbstractDTO]:
        if dataclasses.is_dataclass(model_type):
            return DataclassDTO[model_type]
        elif issubclass(model_type, BaseModel):  # Pydantic
            return PydanticDTO[model_type]
        else:
            return MsgspecDTO[model_type]

# Litestar's factory usage
UserDTO = DTOFactory.create_dto(User)
```

### 47. **Singleton Pattern**

```python
class PluginRegistry:
    """Single instance manages all plugins."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
        return cls._instance
    
    def register(self, plugin):
        self._plugins[plugin.name] = plugin

# Always same instance
registry1 = PluginRegistry()
registry2 = PluginRegistry()
assert registry1 is registry2
```

### 48. **Registry Pattern**

```python
class StoreRegistry:
    """Central registry for stores."""
    
    def __init__(self, stores: dict[str, Store] | None = None):
        self._stores = stores or {}
    
    def register(self, name: str, store: Store) -> None:
        self._stores[name] = store
    
    def get(self, name: str) -> Store:
        return self._stores[name]

# Litestar usage
app.stores.register("cache", RedisStore())
app.stores.register("session", MemoryStore())
```

### 49. **Plugin Architecture**

```python
from typing import Protocol

class PluginProtocol(Protocol):
    """Plugin interface."""
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        ...

class LoggingPlugin:
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        app_config.logging_config = LoggingConfig()
        return app_config

# Registration
app = Litestar(plugins=[LoggingPlugin()])
```

### 50. **Dependency Injection**

```python
from litestar.di import Provide

async def get_database() -> Database:
    return Database()

async def get_user_service(db: Database) -> UserService:
    return UserService(db)

app = Litestar(
    dependencies={
        "db": Provide(get_database),
        "user_service": Provide(get_user_service),
    }
)

@get("/users")
async def get_users(user_service: UserService) -> list[User]:
    # user_service automatically injected
    return await user_service.get_all()
```

## Part 10: Web Framework Concepts

### 51. **Protocol-Based Design**

```python
# Define behavior, not inheritance
class SerializationPlugin(Protocol):
    def is_plugin_supported_type(self, field_definition: FieldDefinition) -> bool:
        ...
    
    def encode_type(self, obj: Any) -> Any:
        ...

# Any class with these methods works
class PydanticPlugin:  # No inheritance!
    def is_plugin_supported_type(self, field_definition):
        return is_pydantic_model(field_definition.annotation)
    
    def encode_type(self, obj):
        return obj.dict()
```

### 52. **Builder Pattern**

```python
class AppBuilder:
    def __init__(self):
        self.routes = []
        self.middleware = []
        self.plugins = []
    
    def add_route(self, route):
        self.routes.append(route)
        return self  # Fluent interface
    
    def add_middleware(self, middleware):
        self.middleware.append(middleware)
        return self
    
    def build(self) -> Litestar:
        return Litestar(
            route_handlers=self.routes,
            middleware=self.middleware,
            plugins=self.plugins
        )

# Usage
app = (AppBuilder()
    .add_route(user_handler)
    .add_middleware(CORSMiddleware())
    .build())
```

### 53. **Decorator Pattern** (Not Python decorators)

```python
class BaseHandler:
    def handle(self, request):
        return Response("Hello")

class CachedHandler:
    def __init__(self, handler):
        self.handler = handler
        self.cache = {}
    
    def handle(self, request):
        if request.path in self.cache:
            return self.cache[request.path]
        response = self.handler.handle(request)
        self.cache[request.path] = response
        return response

# Wrapping behavior
handler = BaseHandler()
cached_handler = CachedHandler(handler)
```

### 54. **Chain of Responsibility (Middleware)**

```python
class Middleware:
    def __init__(self, next_handler):
        self.next = next_handler
    
    async def __call__(self, scope, receive, send):
        # Do something before
        print("Before")
        
        # Call next in chain
        await self.next(scope, receive, send)
        
        # Do something after
        print("After")

# Chain: CORS -> Auth -> RateLimit -> Handler
```

### 55. **Middleware Pipeline**

```python
class CORSMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # Modify headers for CORS
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"access-control-allow-origin", b"*"))
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# Stack middleware
app = handler
app = CORSMiddleware(app)
app = AuthMiddleware(app)
app = CompressionMiddleware(app)
```

### 56. **Route Matching**

```python
# Simple pattern matching
def match_route(path: str, pattern: str) -> dict | None:
    """Match /users/123 against /users/{id}"""
    parts = path.split("/")
    pattern_parts = pattern.split("/")
    
    if len(parts) != len(pattern_parts):
        return None
    
    params = {}
    for part, pattern_part in zip(parts, pattern_parts):
        if pattern_part.startswith("{") and pattern_part.endswith("}"):
            param_name = pattern_part[1:-1]
            params[param_name] = part
        elif part != pattern_part:
            return None
    
    return params

# Litestar uses a trie for efficient matching
```

### 57. **Request/Response Lifecycle**

```python
"""
1. Request arrives
2. ASGI server calls app(scope, receive, send)
3. Middleware stack processes request
4. Route matching finds handler
5. Dependencies injected
6. Handler executed
7. Response serialized
8. Middleware processes response
9. Response sent to client
"""

async def lifecycle(scope, receive, send):
    # 1. Create request object
    request = Request(scope, receive)
    
    # 2. Run before_request hooks
    for hook in app.before_request:
        if result := await hook(request):
            return await send_response(result)
    
    # 3. Match route
    route, params = match_route(request.path)
    
    # 4. Inject dependencies
    kwargs = await inject_dependencies(route.handler)
    
    # 5. Call handler
    response = await route.handler(**kwargs)
    
    # 6. Run after_request hooks
    for hook in app.after_request:
        response = await hook(response)
    
    # 7. Send response
    await send_response(response)
```

### 58. **HTTP Methods**

```python
from enum import Enum

class HttpMethod(str, Enum):
    GET = "GET"      # Retrieve resource
    POST = "POST"    # Create resource
    PUT = "PUT"      # Update (replace) resource
    PATCH = "PATCH"  # Partial update
    DELETE = "DELETE"  # Remove resource
    HEAD = "HEAD"    # GET without body
    OPTIONS = "OPTIONS"  # Get allowed methods

# Method-specific behaviors
STATUS_CODES = {
    HttpMethod.GET: 200,
    HttpMethod.POST: 201,  # Created
    HttpMethod.DELETE: 204,  # No Content
}
```

### 59. **Content Negotiation**

```python
def negotiate_content_type(accept_header: str, available: list[str]) -> str:
    """Choose response format based on Accept header."""
    
    # Parse Accept: application/json, text/html;q=0.9
    for media_type in parse_accept(accept_header):
        if media_type in available:
            return media_type
    
    return available[0]  # Default

@get("/data")
async def get_data(request: Request) -> Response:
    content_type = negotiate_content_type(
        request.headers.get("Accept", "*/*"),
        ["application/json", "text/html"]
    )
    
    if content_type == "text/html":
        return HTMLResponse("<h1>Data</h1>")
    return JSONResponse({"data": "value"})
```

### 60. **CORS (Cross-Origin Resource Sharing)**

```python
class CORSConfig:
    allow_origins: list[str] = ["*"]
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]
    allow_credentials: bool = False
    max_age: int = 600

# Preflight request handling
@options("/{path:path}")
async def handle_preflight(request: Request) -> Response:
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )
```

### 61. **CSRF Protection**

```python
import secrets

class CSRFMiddleware:
    def __init__(self, app, secret: str):
        self.app = app
        self.secret = secret
    
    def generate_token(self) -> str:
        return secrets.token_urlsafe(32)
    
    async def __call__(self, scope, receive, send):
        if scope["method"] in {"POST", "PUT", "DELETE"}:
            # Check CSRF token
            token = get_token_from_request(scope)
            if not self.verify_token(token):
                await send_error(403, "CSRF token invalid")
                return
        
        await self.app(scope, receive, send)
```

### 62. **Rate Limiting**

```python
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, rate: str, limit: int):
        # "minute", 100 = 100 requests per minute
        self.window = {"second": 1, "minute": 60, "hour": 3600}[rate]
        self.limit = limit
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time()
        
        # Clean old requests
        self.requests[key] = [
            t for t in self.requests[key]
            if now - t < self.window
        ]
        
        # Check limit
        if len(self.requests[key]) >= self.limit:
            return False
        
        self.requests[key].append(now)
        return True
```

### 63. **WebSockets**

```python
from litestar import websocket
from litestar.connection import WebSocket

@websocket("/ws")
async def websocket_handler(socket: WebSocket) -> None:
    await socket.accept()
    
    try:
        while True:
            data = await socket.receive_json()
            
            # Process message
            response = process_message(data)
            
            # Send response
            await socket.send_json(response)
    except WebSocketDisconnect:
        pass

# Pub/Sub pattern
@websocket_listener("/chat")
class ChatHandler:
    async def on_connect(self, socket: WebSocket):
        await socket.send_text("Welcome!")
    
    async def on_receive(self, data: str):
        # Broadcast to all
        await socket.send_text(f"Echo: {data}")
    
    async def on_disconnect(self):
        print("User disconnected")
```

## Part 11: Data Transfer & Serialization

### 64. **DTO Pattern**

```python
# Data Transfer Object - separates internal model from API
@dataclass
class UserModel:
    id: int
    username: str
    email: str
    password_hash: str  # Internal only!
    is_admin: bool
    created_at: datetime

# DTO for API responses
class UserResponseDTO(DataclassDTO[UserModel]):
    config = DTOConfig(
        exclude={"password_hash"},  # Never expose
        rename_fields={"username": "name"}  # API uses different name
    )

# DTO for creation
class UserCreateDTO(DataclassDTO[UserModel]):
    config = DTOConfig(
        exclude={"id", "created_at", "is_admin"}  # Server-generated
    )

@post("/users", dto=UserCreateDTO, return_dto=UserResponseDTO)
async def create_user(data: UserModel) -> UserModel:
    # password_hash excluded from input
    # password_hash excluded from output
    return data
```

### 65. **Serialization/Deserialization**

```python
import json
import msgspec

# JSON serialization
data = {"name": "John", "age": 30}
json_str = json.dumps(data)  # Serialize
parsed = json.loads(json_str)  # Deserialize

# msgspec (10x faster)
encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder(type=dict)

encoded = encoder.encode(data)  # bytes
decoded = decoder.decode(encoded)  # dict

# Custom serialization
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
```

### 66. **Schema Validation**

```python
from msgspec import Struct, Meta
from typing import Annotated

class User(Struct):
    name: Annotated[str, Meta(min_length=1, max_length=100)]
    age: Annotated[int, Meta(ge=0, le=150)]
    email: Annotated[str, Meta(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")]

# Validation happens automatically
try:
    user = msgspec.json.decode(b'{"name": "", "age": 200}', type=User)
except ValidationError as e:
    print(e)  # name too short, age too large
```

### 67. **Field Mapping & Renaming**

```python
# Rename strategies
def to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

class DTOConfig:
    rename_strategy: RenameStrategy = "camel"
    rename_fields: dict[str, str] = {
        "internal_name": "externalName"
    }

# Automatic renaming
# Python: user_name, created_at
# JSON: userName, createdAt
```

### 68. **Nested Object Handling**

```python
@dataclass
class Address:
    street: str
    city: str
    country: str

@dataclass
class User:
    name: str
    address: Address  # Nested object
    tags: list[str]  # Collection

# DTO handles nesting automatically
class UserDTO(DataclassDTO[User]):
    config = DTOConfig(
        max_nested_depth=2,  # How deep to serialize
        exclude={"address.country"}  # Exclude nested field
    )

# Recursive handling in Litestar
def transfer_nested(obj, depth=0):
    if depth > max_depth:
        return None
    
    if is_model(obj):
        return {
            field: transfer_nested(value, depth + 1)
            for field, value in obj.__dict__.items()
        }
    return obj
```

## Part 12: Performance Concepts

### 69. **Code Generation**

```python
# Generate optimized code at runtime
def generate_serializer(fields: list[Field]) -> Callable:
    """Generate specialized serialization function."""
    
    code = "def serialize(obj):\n"
    code += "    return {\n"
    
    for field in fields:
        if field.excluded:
            continue
        code += f'        "{field.name}": obj.{field.name},\n'
    
    code += "    }\n"
    
    # Compile and return function
    namespace = {}
    exec(code, namespace)
    return namespace['serialize']

# Generated function is 10x faster than generic
```

### 70. **Caching Strategies**

```python
from functools import lru_cache

# Simple LRU cache
@lru_cache(maxsize=128)
def expensive_computation(x: int) -> int:
    return x ** 2

# Manual caching in Litestar
class DTOBackend:
    _cache: ClassVar[dict] = {}
    
    def get_backend(self, handler_id: str) -> Backend:
        if handler_id not in self._cache:
            self._cache[handler_id] = self.create_backend()
        return self._cache[handler_id]

# Response caching
@get("/data", cache=True)
async def get_data() -> dict:
    # Response cached automatically
    return expensive_query()
```

### 71. **Lazy Loading**

```python
# Import expensive modules only when needed
def get_pydantic_plugin():
    # Lazy import
    try:
        from pydantic import BaseModel
    except ImportError:
        raise MissingDependencyException("Install pydantic")
    
    from litestar.contrib.pydantic import PydanticPlugin
    return PydanticPlugin()

# Lazy attribute
class Litestar:
    @property
    def openapi_schema(self):
        if not hasattr(self, '_openapi_schema'):
            self._openapi_schema = generate_schema()
        return self._openapi_schema
```

### 72. **Memory Optimization**

```python
# Use __slots__ to save memory
class WithSlots:
    __slots__ = ('x', 'y')  # No __dict__
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Memory comparison
import sys
regular = RegularClass()
print(sys.getsizeof(regular.__dict__))  # 296 bytes

slotted = WithSlots(1, 2)
# No __dict__, saves ~40% memory
```

### 73. **Hot Path Optimization**

```python
# Identify and optimize frequently executed code
class Router:
    def match_route(self, path: str):
        # Hot path - called for every request
        # Use trie for O(n) lookup instead of O(m*n)
        
        node = self.trie_root
        for part in path.split('/'):
            if part in node.children:
                node = node.children[part]
            elif node.parameter:
                node = node.parameter
            else:
                return None
        
        return node.handler
```

## Part 13: Libraries and Standards

### 74. **mypy (Static Type Checker)**

```python
# mypy checks types at development time
# Run: mypy your_file.py

def add(a: int, b: int) -> int:
    return a + b

result = add("1", "2")  # mypy error: Expected int, got str

# mypy.ini configuration
"""
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
"""
```

### 75. **pyright (Type Checker)**

```python
# Microsoft's type checker, used in VS Code
# pyrightconfig.json
{
    "include": ["src"],
    "exclude": ["**/node_modules", "**/__pycache__"],
    "strict": ["src/important"],
    "typeCheckingMode": "strict"
}

# Pyright-specific comments
x = 1  # pyright: ignore
reveal_type(x)  # pyright: reveals int
```

### 76. **typing Module**

```python
from typing import (
    Any,       # Any type
    Union,     # Multiple types
    Optional,  # Can be None
    Literal,   # Specific values
    TypeAlias, # Type aliases
    cast,      # Type casting
    overload,  # Multiple signatures
)

# Type alias
JSON: TypeAlias = dict[str, Any]

# Literal types
Mode = Literal["read", "write", "append"]

# Overload
@overload
def process(x: int) -> str: ...

@overload
def process(x: str) -> int: ...

def process(x: int | str) -> str | int:
    if isinstance(x, int):
        return str(x)
    return len(x)
```

### 77. **typing_extensions**

```python
from typing_extensions import (
    Self,          # Reference to current class
    NotRequired,   # Optional TypedDict keys
    TypeGuard,     # Type narrowing
    assert_never,  # Exhaustiveness checking
)

class Node:
    def clone(self) -> Self:
        # Returns same type as class
        return self.__class__()

def is_string(val: object) -> TypeGuard[str]:
    # Narrows type in if statements
    return isinstance(val, str)
```

### 78. **msgspec**

```python
import msgspec
from msgspec import Struct, Meta

# Define structured data
class User(Struct):
    name: str
    age: int = 0
    
# Fast encoding/decoding
encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder(type=User)

data = encoder.encode(User(name="John", age=30))
user = decoder.decode(data)

# 10-50x faster than json module
```

### 79. **Pydantic**

```python
from pydantic import BaseModel, Field, validator

class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# Automatic validation
user = User(name="John", age=30, email="john@example.com")
json_data = user.model_dump_json()
```

### 80. **attrs**

```python
import attr

@attr.s(auto_attribs=True, slots=True)
class User:
    name: str
    age: int = attr.ib(validator=attr.validators.instance_of(int))
    
    @age.validator
    def check_age(self, attribute, value):
        if value < 0:
            raise ValueError("Age must be positive")

# Automatic __init__, __repr__, __eq__
user = User(name="John", age=30)
```

### 81. **JSON/MessagePack**

```python
import json
import msgpack

data = {"name": "John", "age": 30}

# JSON - human readable
json_bytes = json.dumps(data).encode()  # 26 bytes

# MessagePack - binary, smaller, faster
msgpack_bytes = msgpack.packb(data)  # 18 bytes

# Litestar supports both
@get("/data.json")
async def get_json() -> dict:
    return data  # Returns JSON

@get("/data.msgpack", media_type="application/msgpack")
async def get_msgpack() -> dict:
    return data  # Returns MessagePack
```

### 82. **ASGI (Async Server Gateway Interface)**

```python
# ASGI application interface
async def app(scope, receive, send):
    """
    scope: dict with request info
    receive: async callable to get messages
    send: async callable to send messages
    """
    assert scope['type'] == 'http'
    
    # Receive request
    body = b''
    while True:
        message = await receive()
        body += message.get('body', b'')
        if not message.get('more_body', False):
            break
    
    # Send response
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [[b'content-type', b'text/plain']],
    })
    
    await send({
        'type': 'http.response.body',
        'body': b'Hello ASGI!',
    })
```

### 83. **OpenAPI/Swagger**

```python
from litestar.openapi import OpenAPIConfig, OpenAPIController

openapi_config = OpenAPIConfig(
    title="My API",
    version="1.0.0",
    description="API Description",
    servers=[{"url": "https://api.example.com"}],
)

@get(
    "/users/{user_id}",
    summary="Get user by ID",
    description="Retrieves a user by their ID",
    responses={
        200: {"description": "User found"},
        404: {"description": "User not found"}
    }
)
async def get_user(user_id: int) -> User:
    # Automatically documented in OpenAPI
    pass
```

### 84. **HTTP Status Codes**

```python
from http import HTTPStatus

# Common status codes
HTTPStatus.OK  # 200
HTTPStatus.CREATED  # 201
HTTPStatus.NO_CONTENT  # 204
HTTPStatus.BAD_REQUEST  # 400
HTTPStatus.UNAUTHORIZED  # 401
HTTPStatus.FORBIDDEN  # 403
HTTPStatus.NOT_FOUND  # 404
HTTPStatus.INTERNAL_SERVER_ERROR  # 500

# Litestar automatic status codes
@get("/")  # Returns 200
@post("/")  # Returns 201
@delete("/")  # Returns 204
```

### 85. **Content-Type Headers**

```python
from litestar.enums import MediaType

# Common media types
MediaType.JSON  # "application/json"
MediaType.HTML  # "text/html"
MediaType.TEXT  # "text/plain"
MediaType.MESSAGEPACK  # "application/x-msgpack"

@get("/data", media_type=MediaType.JSON)
async def get_data() -> dict:
    return {"key": "value"}
    # Automatically sets Content-Type: application/json
```

## Part 14: Python Internals

### 86. **Metaclasses**

```python
# Metaclass controls class creation
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    pass

# Alternative: __init_subclass__
class PluginBase:
    plugins = []
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.plugins.append(cls)
```

### 87. **Garbage Collection**

```python
import gc
import weakref

# Python uses reference counting + cycle detection
class Node:
    def __init__(self):
        self.ref = None

# Circular reference
a = Node()
b = Node()
a.ref = b
b.ref = a  # Circular!

# Weak references don't prevent GC
a.ref = weakref.ref(b)

# Manual control
gc.collect()  # Force collection
gc.disable()  # Disable automatic GC
gc.enable()   # Re-enable
```

### 88. **Global Interpreter Lock (GIL)**

```python
# GIL prevents true parallelism in threads
import threading

counter = 0

def increment():
    global counter
    for _ in range(1000000):
        counter += 1  # Not thread-safe even with GIL

# Use multiprocessing for CPU-bound tasks
from multiprocessing import Pool

def cpu_bound(x):
    return x ** 2

with Pool() as pool:
    results = pool.map(cpu_bound, range(10))
```

### 89. **Import System**

```python
# Import mechanics
import sys

# Module cache
sys.modules['my_module']  # Cached modules

# Import paths
sys.path.append('/custom/path')

# Lazy imports in Litestar
def get_plugin():
    global PydanticPlugin
    if 'PydanticPlugin' not in globals():
        from litestar.plugins import PydanticPlugin
    return PydanticPlugin()

# Circular import prevention
if TYPE_CHECKING:
    from circular_module import SomeType
```

### 90. **Name Mangling**

```python
class MyClass:
    def __init__(self):
        self.public = 1
        self._protected = 2  # Convention: internal
        self.__private = 3   # Name mangled

obj = MyClass()
print(obj.public)  # Works
print(obj._protected)  # Works but shouldn't use
# print(obj.__private)  # AttributeError
print(obj._MyClass__private)  # Name mangled access
```

## Part 15: Testing & Quality

### 91. **pytest Framework**

```python
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_exception():
    with pytest.raises(ValueError):
        raise ValueError("error")

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected

# Async tests
@pytest.mark.asyncio
async def test_async():
    result = await async_function()
    assert result == expected
```

### 92. **Fixtures**

```python
import pytest
from litestar import Litestar
from litestar.testing import TestClient

@pytest.fixture
def app():
    """Create app for testing."""
    return Litestar(route_handlers=[...])

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app=app)

def test_endpoint(client):
    response = client.get("/users")
    assert response.status_code == 200

# Fixture scope
@pytest.fixture(scope="session")
def database():
    """Shared across all tests."""
    db = create_test_db()
    yield db
    db.cleanup()
```

### 93. **Parametrized Tests**

```python
@pytest.mark.parametrize("method,path,status", [
    ("GET", "/users", 200),
    ("POST", "/users", 201),
    ("GET", "/users/999", 404),
])
def test_routes(client, method, path, status):
    response = client.request(method, path)
    assert response.status_code == status

# Multiple parameters
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [3, 4])
def test_combinations(x, y):
    # Tests: (1,3), (1,4), (2,3), (2,4)
    assert x < y
```

### 94. **Mocking & Patching**

```python
from unittest.mock import Mock, patch

# Mock object
mock_db = Mock()
mock_db.get_user.return_value = User(id=1, name="Test")

# Patch decorator
@patch('module.database')
def test_with_mock(mock_db):
    mock_db.query.return_value = []
    result = function_using_db()
    mock_db.query.assert_called_once()

# Context manager
with patch('module.function') as mock_func:
    mock_func.return_value = "mocked"
    result = call_function()
```

### 95. **Coverage Testing**

```python
# Run with coverage
# pytest --cov=litestar --cov-report=html

# Coverage configuration (pyproject.toml)
"""
[tool.coverage.run]
source = ["litestar"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
"""

# Litestar requires 100% coverage
```

## Part 16: Litestar-Specific Patterns

### 96. **Sentinel Values (Empty vs None)**

```python
from litestar.types import Empty, EmptyType

# Distinguish "not set" from "set to None"
def function(
    value: int | None | EmptyType = Empty
):
    if value is Empty:
        # Not provided at all
        value = get_default()
    elif value is None:
        # Explicitly set to None
        pass
    else:
        # Has a value
        pass

# Used in DTOs
dto: type[AbstractDTO] | None | EmptyType = Empty
```

### 97. **Recursive Type Handling**

```python
def handle_type(type_def, depth=0):
    """Recursively process nested types."""
    
    if depth > MAX_DEPTH:
        return None
    
    if is_collection(type_def):
        inner = get_inner_type(type_def)
        return [handle_type(inner, depth + 1)]
    
    if is_model(type_def):
        return {
            field: handle_type(field_type, depth + 1)
            for field, field_type in get_fields(type_def)
        }
    
    return type_def
```

### 98. **Configuration Layering**

```python
# Configuration hierarchy
app_config = {"timeout": 30, "debug": False}
router_config = {"timeout": 60}  # Override
controller_config = {"cache": True}  # Add
handler_config = {"timeout": 120}  # Override again

# Merged configuration (handler wins)
final_config = {
    "timeout": 120,  # From handler
    "debug": False,  # From app
    "cache": True,   # From controller
}

# Litestar merges automatically
handler.merge(controller, router, app)
```

### 99. **Handler Signature Modeling**

```python
import inspect

def model_handler_signature(handler):
    """Extract parameter information from handler."""
    
    sig = inspect.signature(handler)
    params = {}
    
    for name, param in sig.parameters.items():
        if param.annotation is param.empty:
            continue
            
        params[name] = {
            "type": param.annotation,
            "default": param.default,
            "required": param.default is param.empty
        }
    
    return params

# Litestar uses this for:
# - Dependency injection
# - Parameter validation
# - OpenAPI generation
```

### 100. **Route Reversal (URL Building)**

```python
def route_reverse(name: str, **params) -> str:
    """Build URL from route name and parameters."""
    
    route = get_route_by_name(name)
    path = route.path  # e.g., "/users/{id}/posts/{post_id}"
    
    # Replace parameters
    for param, value in params.items():
        path = path.replace(f"{{{param}}}", str(value))
    
    return path

# Usage
url = app.route_reverse(
    "get_user_post",
    id=123,
    post_id=456
)  # Returns: "/users/123/posts/456"

# Smart type conversion
url = app.route_reverse(
    "get_by_date",
    date="2023-01-01"  # String instead of datetime
)
```

---

## Congratulations! 🎉

You've learned all 100 concepts that power Litestar's sophisticated architecture. These concepts build on each other to create a framework that is:

- **Type-safe**: Extensive use of Python's type system
- **Performant**: Code generation, caching, and optimization
- **Extensible**: Plugin architecture and protocols
- **Developer-friendly**: Automatic validation, serialization, and documentation

Each concept you've learned is actively used in Litestar's codebase. Understanding them will help you:
1. Read and understand Litestar's source code
2. Contribute effectively to the project
3. Build better applications with Litestar
4. Apply these patterns in your own projects

Now you're ready to tackle issues and contribute to Litestar with confidence!