# Python OOP Concepts: Deep Dive for Litestar

## 1. The `__slots__` Magic: Memory Optimization

### The Problem: Python Objects Are Dictionaries

By default, Python stores instance attributes in a dictionary called `__dict__`:

```python
class RegularUser:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

user = RegularUser("Alice", 30, "alice@example.com")
print(user.__dict__)  
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# You can add attributes dynamically!
user.random_attr = "surprise!"  # This works!
print(user.__dict__)
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com', 'random_attr': 'surprise!'}
```

### The Cost

```python
import sys

class RegularClass:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

regular = RegularClass()
print(sys.getsizeof(regular) + sys.getsizeof(regular.__dict__))
# ~296 bytes (varies by Python version)

# The __dict__ alone is expensive!
print(sys.getsizeof(regular.__dict__))  # ~296 bytes
```

### The Solution: `__slots__`

```python
class SlottedUser:
    __slots__ = ('name', 'age', 'email')  # Pre-declare all attributes
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

user = SlottedUser("Alice", 30, "alice@example.com")

# No __dict__!
try:
    print(user.__dict__)
except AttributeError as e:
    print(f"Error: {e}")  # 'SlottedUser' object has no attribute '__dict__'

# Can't add new attributes!
try:
    user.random_attr = "surprise!"
except AttributeError as e:
    print(f"Error: {e}")  # 'SlottedUser' object has no attribute 'random_attr'
```

### Memory Comparison

```python
import sys

class RegularPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class SlottedPoint:
    __slots__ = ('x', 'y', 'z')
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Create many instances
regular_points = [RegularPoint(i, i+1, i+2) for i in range(1000)]
slotted_points = [SlottedPoint(i, i+1, i+2) for i in range(1000)]

# Memory usage (approximate):
# Regular: ~296 bytes per instance = 296KB for 1000 instances
# Slotted: ~64 bytes per instance = 64KB for 1000 instances
# Savings: ~78% less memory!
```

### Performance Benefits

```python
import timeit

class RegularClass:
    def __init__(self):
        self.value = 0

class SlottedClass:
    __slots__ = ('value',)
    
    def __init__(self):
        self.value = 0

# Attribute access is faster with slots!
regular = RegularClass()
slotted = SlottedClass()

# Regular: dictionary lookup
# Slotted: direct memory offset

# Benchmark (your results may vary):
# regular.value access: ~0.07 microseconds
# slotted.value access: ~0.05 microseconds
# ~30% faster attribute access!
```

### How Litestar Uses `__slots__`

```python
# From Litestar's DTOBackend
class DTOBackend:
    __slots__ = (
        "annotation",
        "dto_factory",
        "field_definition",
        "handler_id",
        "is_data_field",
        "model_type",
        "parsed_field_definitions",
        "transfer_model_type",
    )
    
    # Benefits:
    # 1. ~70% less memory per DTO backend
    # 2. Faster attribute access
    # 3. Prevents accidental attribute creation
    # 4. Clear documentation of all attributes
```

### When to Use `__slots__`

```python
# ✅ GOOD: High-performance classes with fixed attributes
class HTTPRequest:
    __slots__ = ('method', 'path', 'headers', 'body', 'query_params')
    
# ✅ GOOD: Classes that will have many instances
class Point3D:
    __slots__ = ('x', 'y', 'z')

# ❌ BAD: Classes that need dynamic attributes
class ConfigurablePlugin:
    # Plugins might need arbitrary attributes
    pass

# ❌ BAD: Classes that will be subclassed often
class BaseModel:
    # Subclasses might need different attributes
    pass
```

## 2. Protocols vs Inheritance: Duck Typing with Types

### Traditional Inheritance: The Rigid Way

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self) -> str:
        pass
    
    @abstractmethod
    def move(self) -> None:
        pass

class Dog(Animal):  # MUST inherit from Animal
    def make_sound(self) -> str:
        return "Woof!"
    
    def move(self) -> None:
        print("Running on four legs")

class Bird(Animal):  # MUST inherit from Animal
    def make_sound(self) -> str:
        return "Tweet!"
    
    def move(self) -> None:
        print("Flying with wings")

def make_animal_sound(animal: Animal) -> str:
    return animal.make_sound()
```

### Problems with Inheritance

```python
# What if we have a Robot that can make sounds but isn't an Animal?
class Robot:  # Doesn't inherit from Animal
    def make_sound(self) -> str:
        return "Beep boop!"
    
    def move(self) -> None:
        print("Rolling on wheels")

robot = Robot()
# make_animal_sound(robot)  # Type error! Robot is not an Animal

# Do we create RobotAnimal? SoundMaker interface? Multiple inheritance mess?
```

### Protocols: Structural Typing (Duck Typing with Safety!)

```python
from typing import Protocol

class SoundMaker(Protocol):
    def make_sound(self) -> str: ...

class Movable(Protocol):
    def move(self) -> None: ...

# No inheritance needed!
class Dog:  # Doesn't inherit from anything
    def make_sound(self) -> str:
        return "Woof!"
    
    def move(self) -> None:
        print("Running")

class Robot:  # Doesn't inherit from anything
    def make_sound(self) -> str:
        return "Beep!"
    
    def move(self) -> None:
        print("Rolling")

class Car:  # Only has move, not make_sound
    def move(self) -> None:
        print("Driving")

def make_noise(thing: SoundMaker) -> str:
    return thing.make_sound()

def start_moving(thing: Movable) -> None:
    thing.move()

# All work without inheritance!
make_noise(Dog())    # ✅ Has make_sound
make_noise(Robot())  # ✅ Has make_sound
# make_noise(Car())  # ❌ Type error: Car doesn't have make_sound

start_moving(Dog())   # ✅ Has move
start_moving(Robot()) # ✅ Has move  
start_moving(Car())   # ✅ Has move
```

### How Litestar Uses Protocols

```python
# Litestar's plugin system uses protocols
from typing import Protocol, Any
from litestar.config.app import AppConfig

class InitPluginProtocol(Protocol):
    """Any class with this method works as a plugin!"""
    def on_app_init(self, app_config: AppConfig) -> AppConfig: ...

class SerializationPluginProtocol(Protocol):
    """Any class with these methods works as a serialization plugin!"""
    def supports_type(self, type_: type) -> bool: ...
    def serialize(self, obj: Any) -> bytes: ...
    def deserialize(self, data: bytes, type_: type) -> Any: ...

# Users can create plugins without inheriting anything!
class MyCustomPlugin:  # No inheritance!
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        print("Initializing my plugin!")
        return app_config

class MySerializer:  # No inheritance!
    def supports_type(self, type_: type) -> bool:
        return type_ == MyCustomType
    
    def serialize(self, obj: Any) -> bytes:
        return str(obj).encode()
    
    def deserialize(self, data: bytes, type_: type) -> Any:
        return type_(data.decode())

# Litestar accepts them because they match the protocol!
app = Litestar(
    plugins=[MyCustomPlugin(), MySerializer()]
)
```

### Protocol vs ABC (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Protocol

# ABC: Inheritance-based
class AbstractSerializer(ABC):
    @abstractmethod
    def serialize(self, data: Any) -> str:
        pass

class JSONSerializer(AbstractSerializer):  # MUST inherit
    def serialize(self, data: Any) -> str:
        return json.dumps(data)

# Protocol: Structure-based
class SerializerProtocol(Protocol):
    def serialize(self, data: Any) -> str: ...

class XMLSerializer:  # No inheritance needed!
    def serialize(self, data: Any) -> str:
        return f"<data>{data}</data>"

def process(serializer: SerializerProtocol) -> None:
    # Works with ANY class that has serialize method!
    result = serializer.serialize({"key": "value"})
```

## 3. Abstract Methods and Classes: Enforcing Contracts

### The Problem: Incomplete Implementations

```python
class PaymentProcessor:
    def process_payment(self, amount: float) -> bool:
        # Oops, someone forgot to implement this!
        pass  # Returns None, not bool!

processor = PaymentProcessor()
result = processor.process_payment(100)  # None! Causes bugs later
```

### Solution: Abstract Base Classes

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """Process payment and return success status."""
        pass
    
    @abstractmethod
    def get_fee(self, amount: float) -> float:
        """Calculate processing fee."""
        pass
    
    # Concrete method that uses abstract methods
    def process_with_fee(self, amount: float) -> tuple[bool, float]:
        fee = self.get_fee(amount)
        total = amount + fee
        success = self.process_payment(total)
        return success, fee

# Can't instantiate abstract class!
try:
    processor = PaymentProcessor()
except TypeError as e:
    print(e)  # Can't instantiate abstract class PaymentProcessor

# Must implement ALL abstract methods
class StripeProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        print(f"Processing ${amount} via Stripe")
        return True
    
    def get_fee(self, amount: float) -> float:
        return amount * 0.029 + 0.30  # 2.9% + 30¢

# Now it works!
stripe = StripeProcessor()
success, fee = stripe.process_with_fee(100)
```

### How Litestar Uses Abstract Methods

```python
# Simplified from Litestar's DTO system
from abc import ABC, abstractmethod
from typing import Any, Generator

class AbstractDTO(ABC):
    """Base DTO that all concrete DTOs must implement."""
    
    @classmethod
    @abstractmethod
    def generate_field_definitions(
        cls, 
        model_type: type[Any]
    ) -> Generator[DTOFieldDefinition, None, None]:
        """Each DTO type must define how to extract fields."""
        ...
    
    @classmethod
    @abstractmethod
    def detect_nested_field(
        cls, 
        field_definition: FieldDefinition
    ) -> bool:
        """Each DTO type must define what counts as nested."""
        ...

# Concrete implementations
class DataclassDTO(AbstractDTO):
    @classmethod
    def generate_field_definitions(cls, model_type):
        # Specific logic for dataclasses
        for field in dataclasses.fields(model_type):
            yield DTOFieldDefinition(...)
    
    @classmethod
    def detect_nested_field(cls, field_definition):
        return dataclasses.is_dataclass(field_definition.annotation)

class PydanticDTO(AbstractDTO):
    @classmethod
    def generate_field_definitions(cls, model_type):
        # Different logic for Pydantic models
        for field_name, field_info in model_type.model_fields.items():
            yield DTOFieldDefinition(...)
    
    @classmethod
    def detect_nested_field(cls, field_definition):
        return issubclass(field_definition.annotation, BaseModel)
```

## 4. Class Methods, Static Methods, and Properties

### Regular Methods vs Class Methods vs Static Methods

```python
class DatabaseConnection:
    _instance = None
    _connection_string = "postgresql://localhost/mydb"
    
    def __init__(self, conn_string):
        self.conn_string = conn_string
        self.connected = False
    
    # Regular method: needs an instance (self)
    def connect(self):
        print(f"Connecting to {self.conn_string}")
        self.connected = True
    
    # Class method: gets the class (cls), not instance
    @classmethod
    def create_default(cls):
        """Factory method to create with default settings."""
        return cls(cls._connection_string)
    
    @classmethod
    def singleton(cls):
        """Singleton pattern using class method."""
        if cls._instance is None:
            cls._instance = cls(cls._connection_string)
        return cls._instance
    
    # Static method: doesn't get self or cls
    @staticmethod
    def validate_connection_string(conn_string: str) -> bool:
        """Pure function, doesn't need class or instance."""
        return conn_string.startswith(("postgresql://", "mysql://"))

# Usage
# Static method: called on class, pure function
valid = DatabaseConnection.validate_connection_string("postgresql://...")

# Class method: called on class, gets class as first arg
default_conn = DatabaseConnection.create_default()
singleton_conn = DatabaseConnection.singleton()

# Regular method: needs instance
default_conn.connect()
```

### Properties: Computed Attributes

```python
class Temperature:
    def __init__(self, celsius: float):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        """Getter for celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        """Setter with validation."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        """Computed property."""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        """Set celsius by setting fahrenheit."""
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self) -> float:
        """Read-only computed property."""
        return self._celsius + 273.15
    
    # No setter for kelvin = read-only!

# Usage looks like attributes!
temp = Temperature(25)
print(temp.celsius)     # 25
print(temp.fahrenheit)  # 77
print(temp.kelvin)      # 298.15

temp.fahrenheit = 86    # Calls setter
print(temp.celsius)     # 30

temp.celsius = -300     # ValueError: Below absolute zero!
temp.kelvin = 0         # AttributeError: can't set attribute (no setter)
```

### How Litestar Uses These

```python
# Simplified from Litestar
class Route:
    def __init__(self, path: str, handler: callable):
        self._path = path
        self._handler = handler
        self._regex = None
    
    @property
    def path(self) -> str:
        """Path is read-only after creation."""
        return self._path
    
    @property
    def regex(self) -> Pattern:
        """Lazy computation and caching."""
        if self._regex is None:
            # Expensive regex compilation happens only once
            self._regex = self._compile_path_regex()
        return self._regex
    
    @classmethod
    def create_from_handler(cls, handler: callable) -> "Route":
        """Factory method extracts path from handler."""
        path = extract_path_from_decorator(handler)
        return cls(path, handler)
    
    @staticmethod
    def is_valid_path(path: str) -> bool:
        """Validation doesn't need instance."""
        return path.startswith("/") and "{" not in path or "}" not in path
```

## 5. Metaclasses and `__class_getitem__`: Class Factories

### What Are Metaclasses?

```python
# Everything in Python is an object, even classes!
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
print(type(type))     # <class 'type'>

# 'type' is the metaclass - the class that creates classes!

# You can create classes dynamically:
def init(self, name):
    self.name = name

# Creating a class with type()
DynamicClass = type(
    'DynamicClass',           # Class name
    (),                       # Base classes
    {'__init__': init, 'x': 5}  # Class dict
)

instance = DynamicClass("test")
print(instance.name)  # "test"
print(instance.x)     # 5
```

### Custom Metaclasses

```python
class SingletonMeta(type):
    """Metaclass that creates singleton classes."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        # __call__ on metaclass controls instance creation
        if cls not in cls._instances:
            # Create instance only once
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"Creating database connection to {connection_string}")

# Only creates one instance!
db1 = Database("postgresql://localhost")  # Prints: Creating database...
db2 = Database("postgresql://localhost")  # No print!
db3 = Database("mysql://localhost")       # No print!

print(db1 is db2 is db3)  # True - all same instance!
```

### The `__class_getitem__` Magic

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, items: list[T]):
        self.items = items
    
    @classmethod
    def __class_getitem__(cls, item):
        """Called when you write Container[int]."""
        print(f"Creating Container specialized for {item}")
        
        # Create a new class specialized for this type
        class SpecializedContainer(cls):
            item_type = item
            
            def validate(self, value):
                if not isinstance(value, self.item_type):
                    raise TypeError(f"Expected {self.item_type}")
        
        return SpecializedContainer

# When you write this:
IntContainer = Container[int]
# Python calls: Container.__class_getitem__(int)

int_container = IntContainer([1, 2, 3])
int_container.validate(5)    # OK
int_container.validate("5")  # TypeError!
```

### How Litestar Uses `__class_getitem__`

```python
# Simplified from Litestar's DTO system
class AbstractDTO(Generic[T]):
    model_type: type[T] = None
    
    @classmethod
    def __class_getitem__(cls, model_type: type[T]) -> type["AbstractDTO[T]"]:
        """Create a DTO specialized for a model type."""
        
        # Extract type information
        if hasattr(model_type, "__args__"):
            # Handle Generic types like List[User]
            actual_type = model_type.__origin__
            type_args = model_type.__args__
        else:
            actual_type = model_type
            type_args = None
        
        # Create specialized DTO class
        dto_class = type(
            f"{cls.__name__}[{actual_type.__name__}]",
            (cls,),
            {
                "model_type": actual_type,
                "type_args": type_args,
            }
        )
        
        return dto_class

# Usage:
@dataclass
class User:
    name: str
    age: int

# This creates a new class!
UserDTO = DataclassDTO[User]
print(UserDTO.model_type)  # <class 'User'>

# Different from:
ProductDTO = DataclassDTO[Product]
print(UserDTO is ProductDTO)  # False - different classes!
```

## 6. Multiple Inheritance and MRO (Method Resolution Order)

### The Diamond Problem

```python
class A:
    def method(self):
        print("A's method")

class B(A):
    def method(self):
        print("B's method")

class C(A):
    def method(self):
        print("C's method")

class D(B, C):  # Multiple inheritance!
    pass

# Which method gets called?
d = D()
d.method()  # "B's method" - but why?

# Python uses C3 linearization for MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# Searches left-to-right: D -> B -> C -> A -> object
```

### Cooperative Multiple Inheritance with `super()`

```python
class TimestampMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Cooperative!
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update(self):
        self.updated_at = datetime.now()
        super().update()  # Call next in MRO

class ValidatorMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Cooperative!
        self.is_valid = False
    
    def validate(self):
        self.is_valid = True
        return self.is_valid

class Model:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class User(TimestampMixin, ValidatorMixin, Model):
    def __init__(self, name, email):
        super().__init__(name=name, email=email)

user = User("Alice", "alice@example.com")
print(user.created_at)  # From TimestampMixin
print(user.is_valid)     # From ValidatorMixin  
print(user.name)         # From Model
```

### How Litestar Uses Multiple Inheritance

```python
# Simplified pattern from Litestar
class RouterMixin:
    """Provides routing functionality."""
    def add_route(self, path: str, handler: callable):
        self.routes[path] = handler

class MiddlewareMixin:
    """Provides middleware functionality."""
    def add_middleware(self, middleware: callable):
        self.middleware.append(middleware)

class ConfigMixin:
    """Provides configuration functionality."""
    def configure(self, **options):
        self.config.update(options)

class Litestar(RouterMixin, MiddlewareMixin, ConfigMixin):
    """Main app class combines all functionality."""
    def __init__(self):
        self.routes = {}
        self.middleware = []
        self.config = {}
        super().__init__()  # Initialize all mixins

app = Litestar()
app.add_route("/", handler)         # From RouterMixin
app.add_middleware(cors_middleware)  # From MiddlewareMixin
app.configure(debug=True)           # From ConfigMixin
```

## 7. Context Managers and Async Context Managers

### Basic Context Managers

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"Connecting to {self.connection_string}")
        self.connection = create_connection(self.connection_string)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block (even on exception!)."""
        print("Closing connection")
        if self.connection:
            self.connection.close()
        # Return False to propagate exceptions

# Usage:
with DatabaseConnection("postgresql://localhost") as conn:
    conn.execute("SELECT * FROM users")
    # Even if this fails, connection closes!
# Connection automatically closed here
```

### Async Context Managers

```python
class AsyncDatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        """Async version of __enter__."""
        print(f"Connecting to {self.connection_string}")
        self.connection = await create_async_connection(self.connection_string)
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async version of __exit__."""
        print("Closing connection")
        if self.connection:
            await self.connection.close()

# Usage:
async with AsyncDatabaseConnection("postgresql://localhost") as conn:
    await conn.execute("SELECT * FROM users")
# Connection automatically closed here
```

### How Litestar Uses AsyncExitStack

```python
from contextlib import AsyncExitStack

class Litestar:
    async def lifespan(self):
        """Complex lifecycle management."""
        async with AsyncExitStack() as stack:
            # Register cleanup in REVERSE order
            for hook in reversed(self.on_shutdown):
                stack.push_async_callback(hook)
            
            # Start services
            db = await stack.enter_async_context(
                DatabaseConnection()
            )
            cache = await stack.enter_async_context(
                CacheConnection()
            )
            
            # Run startup hooks
            for hook in self.on_startup:
                await hook()
            
            yield  # App runs here
            
            # Everything cleaned up automatically in reverse order!
            # 1. Shutdown hooks run
            # 2. Cache closes
            # 3. Database closes
```

## Real-World Example: How It All Comes Together in Litestar

```python
from abc import ABC, abstractmethod
from typing import Protocol, Generic, TypeVar, ClassVar
from contextlib import asynccontextmanager

T = TypeVar('T')

# Protocol for plugins
class PluginProtocol(Protocol):
    def on_startup(self) -> None: ...

# Abstract base for DTOs
class AbstractDTO(ABC, Generic[T]):
    __slots__ = ('_model_type', '_config')  # Memory optimization
    _instances: ClassVar[dict] = {}  # Shared cache
    
    @classmethod
    @abstractmethod
    def parse(cls, data: dict) -> T:
        """Must be implemented by subclasses."""
        ...
    
    @classmethod
    def __class_getitem__(cls, model_type: type[T]) -> type["AbstractDTO[T]"]:
        """Create specialized DTO classes."""
        if model_type not in cls._instances:
            cls._instances[model_type] = type(
                f"{cls.__name__}[{model_type.__name__}]",
                (cls,),
                {"_model_type": model_type}
            )
        return cls._instances[model_type]
    
    @property
    def model_type(self) -> type[T]:
        """Computed property."""
        return self._model_type

# Multiple inheritance for features
class TimestampMixin:
    created_at: datetime
    updated_at: datetime

class ValidatedMixin:
    @abstractmethod
    def validate(self) -> bool: ...

# Main app using everything
class Litestar(TimestampMixin, ValidatedMixin):
    __slots__ = ('routes', 'plugins', '_started')
    
    def __init__(self):
        self.routes = {}
        self.plugins: list[PluginProtocol] = []
        self._started = False
        super().__init__()
    
    @property
    def is_running(self) -> bool:
        return self._started
    
    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for lifecycle."""
        try:
            # Startup
            for plugin in self.plugins:
                plugin.on_startup()
            self._started = True
            yield self
        finally:
            # Shutdown (always runs!)
            self._started = False
    
    def validate(self) -> bool:
        """Implementation of abstract method."""
        return len(self.routes) > 0

# Usage showcasing all concepts:
async def main():
    app = Litestar()
    
    # Property access
    print(app.is_running)  # False
    
    # Async context manager
    async with app.lifespan() as running_app:
        print(running_app.is_running)  # True
        
        # Use specialized DTO
        UserDTO = AbstractDTO[User]
        user = UserDTO.parse({"name": "Alice"})
    
    print(app.is_running)  # False (cleaned up!)
```

## Key Takeaways

1. **`__slots__`**: Trade flexibility for performance (70% less memory, 30% faster access)
2. **Protocols**: Duck typing with type safety (no inheritance needed!)
3. **Abstract Classes**: Enforce implementation contracts
4. **Properties**: Computed attributes with validation
5. **Class/Static Methods**: Different scopes for different needs
6. **Metaclasses**: Classes that create classes
7. **`__class_getitem__`**: Dynamic class specialization (how Generic works!)
8. **Context Managers**: Guaranteed cleanup, even on errors
9. **Multiple Inheritance**: Combine functionalities with MRO

Litestar uses ALL these patterns to achieve:
- **Performance**: Slots for memory, properties for caching
- **Flexibility**: Protocols for plugins, metaclasses for DTOs
- **Safety**: Abstract classes for contracts, context managers for cleanup
- **Type Safety**: Generic specialization, proper inheritance hierarchies