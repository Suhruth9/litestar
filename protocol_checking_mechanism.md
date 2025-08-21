# How Protocol Checking Actually Works

## The Key: Type Annotations Tell the Story

Litestar (and Python) knows which Protocol to check based on **type annotations in function signatures**. Let me break this down:

## 1. Type Annotations Are the Trigger

```python
from typing import Protocol

class CacheProtocol(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes) -> None: ...

class AuthProtocol(Protocol):
    def authenticate(self, token: str) -> User: ...

# The type annotation tells which Protocol to check!
def use_cache(cache: CacheProtocol) -> None:  # ← This annotation!
    data = cache.get("key")

def use_auth(auth: AuthProtocol) -> None:     # ← This annotation!
    user = auth.authenticate("token")

# Type checker knows:
# - First function needs CacheProtocol
# - Second function needs AuthProtocol
```

## 2. How Litestar Uses This Pattern

```python
# In Litestar's codebase (simplified)
from typing import Protocol, Any

class SerializationPlugin(Protocol):
    def supports_type(self, type_: type) -> bool: ...
    def serialize(self, obj: Any) -> bytes: ...

class Litestar:
    def __init__(
        self,
        plugins: list[SerializationPlugin] = None  # ← Type annotation!
    ):
        self.plugins = plugins or []
    
    def serialize_response(
        self, 
        data: Any,
        plugin: SerializationPlugin  # ← Type annotation!
    ) -> bytes:
        if plugin.supports_type(type(data)):
            return plugin.serialize(data)

# When you pass something:
class MySerializer:
    def supports_type(self, type_: type) -> bool:
        return True
    def serialize(self, obj: Any) -> bytes:
        return str(obj).encode()

app = Litestar(plugins=[MySerializer()])  
# Type checker sees: Does MySerializer match SerializationPlugin?
# Checks: Does it have supports_type? Yes ✓
# Checks: Does it have serialize? Yes ✓
# Result: Valid!
```

## 3. The Two-Phase Process

### Phase 1: Type Checking Time (mypy/pyright)
```python
# This happens when you run: mypy your_code.py

class RedisCache:
    def get(self, key: str) -> bytes | None:
        return redis.get(key)
    def set(self, key: str, value: bytes) -> None:
        redis.set(key, value)

def configure_cache(cache: CacheProtocol) -> None:
    cache.set("test", b"data")

# Type checker analyzes:
configure_cache(RedisCache())  
# 1. Function expects CacheProtocol
# 2. RedisCache has get(str) -> bytes | None? ✓
# 3. RedisCache has set(str, bytes) -> None? ✓
# 4. PASSES type checking!

configure_cache(dict())  
# 1. Function expects CacheProtocol
# 2. dict has get(str) -> bytes | None? ✗ (different signature)
# 3. FAILS type checking!
```

### Phase 2: Runtime (Python execution)
```python
# This happens when you run: python your_code.py

def configure_cache(cache: CacheProtocol) -> None:
    # At runtime, Python doesn't check protocols!
    # It just calls the methods
    cache.set("test", b"data")  # Direct method call

# This works at runtime even without proper methods!
class FakeCache:
    def set(self, key, value):
        print(f"Setting {key}")

configure_cache(FakeCache())  # Works at runtime!

# But this crashes:
class BrokenCache:
    pass

configure_cache(BrokenCache())  # AttributeError at runtime!
```

## 4. How Litestar Determines Which Protocol

### Through Function Parameters
```python
# Litestar's internal code pattern:
class Litestar:
    def add_plugin(self, plugin: InitPluginProtocol) -> None:
        # Type annotation declares: "I need InitPluginProtocol"
        plugin.on_app_init(self.config)
    
    def add_serializer(self, serializer: SerializationPlugin) -> None:
        # Type annotation declares: "I need SerializationPlugin"
        self.serializers.append(serializer)
    
    def add_cache(self, cache: CacheProtocol) -> None:
        # Type annotation declares: "I need CacheProtocol"
        self.cache_backend = cache

# User code:
app.add_plugin(MyPlugin())      # Checked against InitPluginProtocol
app.add_serializer(MySerializer())  # Checked against SerializationPlugin
app.add_cache(RedisCache())      # Checked against CacheProtocol
```

### Through Type Parameters
```python
# Generic protocols with type parameters
from typing import Generic, TypeVar

T = TypeVar('T')

class Repository(Protocol, Generic[T]):
    def get(self, id: int) -> T: ...
    def save(self, item: T) -> None: ...

def process_users(repo: Repository[User]) -> None:
    # Type checker knows: repo must have get/save for User type
    user = repo.get(1)  # Returns User
    repo.save(user)

def process_products(repo: Repository[Product]) -> None:
    # Type checker knows: repo must have get/save for Product type
    product = repo.get(1)  # Returns Product
    repo.save(product)
```

## 5. The Magic: `isinstance()` with Protocols (Python 3.8+)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable  # Special decorator!
class CacheProtocol(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes) -> None: ...

class MyCache:
    def get(self, key: str) -> bytes | None:
        return None
    def set(self, key: str, value: bytes) -> None:
        pass

# Now you can check at runtime!
cache = MyCache()
print(isinstance(cache, CacheProtocol))  # True!

# But be careful - it only checks method names, not signatures!
class BadCache:
    def get(self): pass  # Wrong signature!
    def set(self): pass  # Wrong signature!

print(isinstance(BadCache(), CacheProtocol))  # Still True! (only checks names)
```

## 6. How Litestar Really Uses This

```python
# Simplified from Litestar's actual code
from typing import Protocol, Union

class InitPluginProtocol(Protocol):
    def on_app_init(self, app_config: AppConfig) -> AppConfig: ...

class SerializationPluginProtocol(Protocol):
    def supports_type(self, type_: type) -> bool: ...
    def serialize(self, obj: Any) -> bytes: ...

# Union type for multiple protocols
PluginType = Union[InitPluginProtocol, SerializationPluginProtocol]

class Litestar:
    def __init__(self, plugins: list[PluginType] = None):
        self.init_plugins: list[InitPluginProtocol] = []
        self.serialization_plugins: list[SerializationPluginProtocol] = []
        
        for plugin in (plugins or []):
            # Check which protocol it matches (at runtime using hasattr)
            if hasattr(plugin, 'on_app_init'):
                self.init_plugins.append(plugin)
            if hasattr(plugin, 'supports_type'):
                self.serialization_plugins.append(plugin)
```

## 7. The Dependency Injection Case

```python
# Litestar's DI system uses type annotations
from litestar import Litestar, get
from litestar.di import Provide

class DatabaseProtocol(Protocol):
    async def query(self, sql: str) -> list[dict]: ...

class CacheProtocol(Protocol):
    async def get(self, key: str) -> Any: ...

async def get_database() -> DatabaseProtocol:
    return PostgresDatabase()

async def get_cache() -> CacheProtocol:
    return RedisCache()

@get("/users")
async def get_users(
    db: DatabaseProtocol,    # ← Type annotation triggers DI
    cache: CacheProtocol,   # ← Type annotation triggers DI
) -> list[User]:
    # Litestar sees these annotations and knows:
    # 1. db needs DatabaseProtocol - call get_database()
    # 2. cache needs CacheProtocol - call get_cache()
    
    cached = await cache.get("users")
    if cached:
        return cached
    
    users = await db.query("SELECT * FROM users")
    return users

app = Litestar(
    route_handlers=[get_users],
    dependencies={
        DatabaseProtocol: Provide(get_database),
        CacheProtocol: Provide(get_cache),
    }
)
```

## The Complete Picture

```python
# 1. DECLARATION: Define what you need
class PluginProtocol(Protocol):
    def process(self, data: Any) -> Any: ...

# 2. ANNOTATION: Say where you need it
def use_plugin(plugin: PluginProtocol) -> None:  # ← The key!
    result = plugin.process("data")

# 3. TYPE CHECKING: Validates at development time
class MyPlugin:
    def process(self, data: Any) -> Any:
        return data.upper()

use_plugin(MyPlugin())  # Type checker validates this

# 4. RUNTIME: Just runs the code
# Python doesn't enforce protocols at runtime
# It just calls the methods and hopes they exist!
```

## Summary: How Does Litestar Know?

1. **Type Annotations**: Function parameters declare which Protocol they expect
2. **Type Checker**: Validates that passed objects match the Protocol (development time)
3. **Runtime**: Python just calls the methods (no Protocol checking unless @runtime_checkable)
4. **Dependency Injection**: Litestar reads annotations to determine what to inject
5. **Registration**: When you register plugins/components, their type annotations tell Litestar how to use them

The key insight: **The type annotation is the declaration of intent**. When you write `cache: CacheProtocol`, you're telling both the type checker AND Litestar: "I need something that matches CacheProtocol here."