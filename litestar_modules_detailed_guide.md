# Litestar Modules - Detailed Contributor Guide

## Module Importance Tiers

### Tier 1: Core Essential (Must understand to contribute)
1. **app.py** - Application core
2. **handlers/** - Request handling  
3. **router.py** - Routing system
4. **connection/** - Request/Response abstractions
5. **dto/** - Data transformation

### Tier 2: Important (Frequently modified)
6. **middleware/** - Request pipeline
7. **plugins/** - Extension system
8. **response/** - Response types
9. **datastructures/** - Core data types
10. **config/** - Configuration

### Tier 3: Specialized (Domain-specific contributions)
11. **contrib/** - Third-party integrations
12. **openapi/** - API documentation
13. **security/** - Authentication
14. **testing/** - Test utilities
15. **cli/** - Command-line interface

### Tier 4: Supporting (Rarely modified directly)
16. **types/** - Type definitions
17. **utils/** - Utilities
18. **exceptions/** - Error handling
19. **events/** - Event system
20. **serialization/** - Data serialization

---

## Detailed Module Breakdown

## 1. `app.py` - The Heart of Litestar
**Complexity**: ⭐⭐⭐⭐⭐ Complex  
**Lines**: ~1,042  
**Critical for**: Any core framework changes

### What it does:
The `Litestar` class is the main application instance that orchestrates the entire framework. It's like the conductor of an orchestra, coordinating all other components.

### Key Responsibilities:
```python
# Main application class structure
class Litestar(Router):
    def __init__(
        self,
        route_handlers=None,
        middleware=None,
        plugins=None,
        dependencies=None,
        exception_handlers=None,
        on_app_init=None,
        # ... many more parameters
    ):
        # Initializes router
        # Registers plugins
        # Sets up middleware stack
        # Configures OpenAPI
        # Manages application lifecycle
```

### Key Methods:
- `__init__()`: Sets up the entire application configuration
- `__call__()`: Makes the app ASGI-callable
- `register()`: Dynamically registers handlers/controllers
- `emit()`: Event system for lifecycle hooks

### Common Contribution Areas:
- Adding new application-level configuration options
- Implementing new lifecycle hooks
- Improving plugin initialization
- Performance optimizations in request dispatch

### Important Files to Understand:
- `litestar/app.py:Litestar` - Main class
- `litestar/config/app.py:AppConfig` - Configuration structure

---

## 2. `handlers/` - Request Processing Engine
**Complexity**: ⭐⭐⭐⭐⭐ Complex  
**Files**: 13  
**Critical for**: Adding new decorators, handler types

### Module Structure:
```
handlers/
├── http_handlers/
│   ├── decorators.py     # @get, @post, etc. (1,253 lines!)
│   └── base.py           # HTTPRouteHandler class (800 lines)
├── websocket_handlers/
│   ├── listener.py       # WebSocket event handling (548 lines)
│   └── stream.py         # WebSocket streaming
└── asgi_handlers.py      # Raw ASGI handlers
```

### HTTP Handler Example:
```python
from litestar import get, post

@get("/users/{user_id:int}")
async def get_user(user_id: int) -> User:
    # Handler function with type hints
    # Litestar automatically:
    # - Extracts user_id from path
    # - Validates it's an integer
    # - Serializes User to JSON
    return await db.get_user(user_id)
```

### WebSocket Handler Types:
```python
# 1. Basic WebSocket
@websocket("/ws")
async def websocket_handler(socket: WebSocket) -> None:
    await socket.accept()
    await socket.send_text("Hello")
    
# 2. Listener Pattern (event-based)
@websocket_listener("/events")
async def handle_events(data: str) -> str:
    return f"Echo: {data}"
    
# 3. Stream Pattern (generator-based)
@websocket_stream("/stream")
async def stream_data(socket: WebSocket) -> AsyncGenerator[str, None]:
    while True:
        yield await get_next_message()
```

### Key Concepts:
- **Signature Modeling**: Analyzes function signatures to understand parameters
- **Dependency Injection**: Automatic parameter resolution
- **Layered Configuration**: Handler > Controller > Router > App level configs
- **Before/After Hooks**: Request/response processing hooks

### Common Contributions:
- New handler decorators
- Performance improvements in signature parsing
- WebSocket protocol enhancements
- Handler validation improvements

---

## 3. `router.py` - URL Routing System
**Complexity**: ⭐⭐⭐ Moderate  
**Critical for**: Route organization features

### What it does:
Groups handlers and controllers hierarchically, like a file system for URLs.

### Router Hierarchy Example:
```python
from litestar import Router, get

# Sub-router for API v1
api_v1_router = Router(
    path="/api/v1",
    route_handlers=[users_controller, posts_controller],
    middleware=[APIKeyMiddleware],
    dependencies={"db": provide_database}
)

# Sub-router for API v2
api_v2_router = Router(
    path="/api/v2",
    route_handlers=[new_users_controller],
    middleware=[JWTMiddleware]
)

# Main app combines routers
app = Litestar(
    route_handlers=[api_v1_router, api_v2_router, health_check],
    middleware=[CORSMiddleware]
)
```

### Key Features:
- **Path Prefixing**: Automatic path concatenation
- **Middleware Layering**: Router-specific middleware
- **Dependency Scoping**: Router-level dependencies
- **Handler Registration**: Collects and organizes handlers

---

## 4. `dto/` - Data Transfer Objects
**Complexity**: ⭐⭐⭐⭐⭐ Complex  
**Files**: 11  
**Critical for**: Data validation/serialization features

### What it does:
DTOs transform data between internal models and API representations, providing security, validation, and performance.

### DTO Architecture:
```python
# 1. Define your model
@dataclass
class UserModel:
    id: int
    username: str
    email: str
    password_hash: str  # Sensitive!
    created_at: datetime
    is_admin: bool

# 2. Create DTOs for different use cases
class UserCreateDTO(DataclassDTO[UserModel]):
    config = DTOConfig(
        exclude={"id", "created_at", "is_admin", "password_hash"},
        rename_fields={"username": "user_name"}
    )

class UserResponseDTO(DataclassDTO[UserModel]):
    config = DTOConfig(
        exclude={"password_hash"},  # Never expose password
        max_nested_depth=2
    )

# 3. Use in handlers
@post("/users", dto=UserCreateDTO, return_dto=UserResponseDTO)
async def create_user(data: UserModel) -> UserModel:
    # data only has fields allowed by UserCreateDTO
    # response automatically filtered by UserResponseDTO
    return await save_user(data)
```

### Key Files:
- `dto/_backend.py` - Core DTO implementation (969 lines)
- `dto/_codegen_backend.py` - Code generation for performance (596 lines)
- `dto/dataclass_dto.py` - Dataclass support
- `dto/msgspec_dto.py` - Msgspec integration

### How DTOs Work:
1. **Parse Model**: Analyzes the data model structure
2. **Generate Code**: Creates optimized serialization functions
3. **Cache Functions**: Stores generated code for reuse
4. **Transform Data**: Applies inclusion/exclusion/renaming rules

### Performance Magic:
```python
# DTOs generate specialized code like:
def serialize_user(obj):
    return {
        "id": obj.id,
        "user_name": obj.username,  # Renamed field
        "email": obj.email,
        "created_at": obj.created_at.isoformat()
        # password_hash excluded
    }
# This is 10-50x faster than generic serialization
```

---

## 5. `connection/` - Request/Response Abstractions
**Complexity**: ⭐⭐⭐ Moderate  
**Files**: 4  
**Critical for**: Request handling features

### What it does:
Provides high-level abstractions over ASGI connection scope, making it easy to work with requests and WebSockets.

### Request Object:
```python
from litestar import Request, get

@get("/info")
async def handler(request: Request) -> dict:
    # Request provides easy access to:
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "cookies": request.cookies,
        "client": request.client,
        "user": request.user,  # From auth middleware
        "state": request.state  # Request-scoped storage
    }
```

### WebSocket Connection:
```python
from litestar import WebSocket, websocket

@websocket("/ws")
async def websocket_handler(socket: WebSocket) -> None:
    await socket.accept()
    
    # Receive different data types
    text = await socket.receive_text()
    bytes_data = await socket.receive_bytes()
    json_data = await socket.receive_json()
    
    # Send responses
    await socket.send_text("Hello")
    await socket.send_json({"status": "connected"})
    
    # Connection state
    if socket.client:
        print(f"Client: {socket.client}")
```

---

## 6. `middleware/` - Request/Response Pipeline
**Complexity**: ⭐⭐⭐⭐ Complex  
**Files**: 24  
**Critical for**: Adding processing layers

### What it does:
Processes requests before handlers and responses after handlers, like a series of filters.

### Middleware Types:

#### Built-in Middleware:
```python
from litestar import Litestar
from litestar.middleware import (
    CORSMiddleware,
    CSRFMiddleware, 
    RateLimitMiddleware,
    CompressionMiddleware,
    AuthenticationMiddleware
)

app = Litestar(
    middleware=[
        CORSMiddleware(allow_origins=["https://example.com"]),
        RateLimitMiddleware(rate_limit=("minute", 100)),
        CompressionMiddleware(minimum_size=1000),
    ]
)
```

#### Custom Middleware:
```python
from litestar import Request, Response
from litestar.middleware import AbstractMiddleware

class TimingMiddleware(AbstractMiddleware):
    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send
    ) -> None:
        start = time.time()
        
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Add timing header
                headers = message.get("headers", [])
                duration = time.time() - start
                headers.append((b"x-response-time", f"{duration:.3f}".encode()))
                message["headers"] = headers
            await send(message)
            
        await self.app(scope, receive, send_wrapper)
```

### Middleware Execution Order:
```
Request → CORS → Auth → RateLimit → Compression → Handler
        ↓                                            ↓
Response ← CORS ← Auth ← RateLimit ← Compression ← Handler
```

---

## 7. `plugins/` - Extension System
**Complexity**: ⭐⭐⭐⭐ Complex  
**Files**: 21  
**Critical for**: Framework extensions

### Plugin Types:

#### 1. InitPlugin - Modify app at startup:
```python
from litestar.plugins import InitPluginProtocol

class DatabasePlugin(InitPluginProtocol):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        # Add database to dependencies
        app_config.dependencies["db"] = self.provide_database
        # Add shutdown handler
        app_config.on_shutdown.append(self.close_database)
        return app_config
```

#### 2. SerializationPlugin - Custom types:
```python
class MongoDBPlugin(SerializationPluginProtocol):
    def is_plugin_supported_type(self, field_definition: FieldDefinition) -> bool:
        return issubclass(field_definition.annotation, ObjectId)
        
    def encode_type(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        return obj
```

#### 3. OpenAPISchemaPlugin - Schema customization:
```python
class CustomSchemaPlugin(OpenAPISchemaPluginProtocol):
    def is_plugin_supported_field(self, field_definition: FieldDefinition) -> bool:
        return field_definition.annotation is CustomType
        
    def to_openapi_schema(self, field_definition: FieldDefinition) -> Schema:
        return Schema(
            type="string",
            format="custom",
            description="Custom type field"
        )
```

---

## 8. `contrib/` - Third-Party Integrations
**Complexity**: ⭐⭐⭐⭐ Complex  
**Files**: 58 (largest module!)  
**Critical for**: External library support

### Major Integrations:

#### SQLAlchemy Integration:
```python
from litestar.contrib.sqlalchemy import SQLAlchemyPlugin, SQLAlchemyDTO

plugin = SQLAlchemyPlugin(
    connection_string="postgresql://localhost/db",
    session_dependency_key="session",
    engine_dependency_key="engine"
)

class UserDTO(SQLAlchemyDTO[User]):
    config = DTOConfig(exclude={"password_hash"})

@get("/users", return_dto=UserDTO)
async def get_users(session: AsyncSession) -> list[User]:
    return await session.scalars(select(User))
```

#### JWT Authentication:
```python
from litestar.contrib.jwt import JWTAuth, Token

jwt_auth = JWTAuth[User](
    retrieve_user_handler=get_user_by_id,
    token_secret="secret-key",
    exclude=["/login", "/register"]
)

@post("/login")
async def login(data: LoginData) -> Response[Token]:
    user = await authenticate(data)
    return jwt_auth.create_token(user=user)
```

#### Pydantic Support:
```python
from litestar.contrib.pydantic import PydanticPlugin, PydanticDTO

class UserModel(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(gt=0, le=150)

@post("/users", dto=PydanticDTO[UserModel])
async def create_user(data: UserModel) -> UserModel:
    return data
```

---

## 9. `openapi/` - API Documentation
**Complexity**: ⭐⭐⭐⭐ Complex  
**Files**: 38  
**Critical for**: API documentation features

### What it does:
Automatically generates OpenAPI 3.1 specification from your code.

### Features:
```python
from litestar import get, post
from litestar.openapi import OpenAPIConfig, OpenAPIController

# Configure OpenAPI
openapi_config = OpenAPIConfig(
    title="My API",
    version="1.0.0",
    description="API for my application",
    servers=[{"url": "https://api.example.com"}],
    external_docs={"url": "https://docs.example.com"},
    tags=[
        {"name": "users", "description": "User operations"},
        {"name": "posts", "description": "Post operations"}
    ]
)

# Handlers automatically documented
@get(
    "/users/{user_id:int}",
    tags=["users"],
    summary="Get user by ID",
    description="Retrieves a user by their unique identifier",
    responses={
        200: {"description": "User found"},
        404: {"description": "User not found"}
    }
)
async def get_user(user_id: int) -> User:
    ...

app = Litestar(
    route_handlers=[...],
    openapi_config=openapi_config
)
```

### Key Components:
- **spec/** - OpenAPI specification models
- **controller.py** - Serves OpenAPI JSON and UI
- **parameters.py** - Parameter documentation
- **responses.py** - Response documentation

---

## 10. `middleware/` Deep Dive

### CORS Middleware:
```python
CORSMiddleware(
    allow_origins=["https://frontend.com"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
    expose_headers=["X-Total-Count"],
    max_age=3600
)
```

### Rate Limiting:
```python
from litestar.middleware.rate_limit import RateLimitMiddleware

RateLimitMiddleware(
    rate_limit=("minute", 100),  # 100 requests per minute
    exclude=["/health"],  # Don't rate limit health checks
    check_throttle_handler=custom_throttle_check  # Custom logic
)
```

### Authentication Middleware:
```python
from litestar.middleware.authentication import AuthenticationMiddleware

class CustomAuth(AuthenticationMiddleware):
    async def authenticate_request(
        self, connection: ASGIConnection
    ) -> AuthenticationResult:
        token = connection.headers.get("Authorization")
        if token:
            user = await validate_token(token)
            return AuthenticationResult(user=user, auth=token)
        return AuthenticationResult()
```

---

## Contributing Guidelines by Module Type

### For Core Modules (app, handlers, router):
1. **Deep understanding required** - Changes affect entire framework
2. **Extensive testing needed** - Break here, break everywhere
3. **Performance critical** - Benchmark before/after changes
4. **Backward compatibility** - Don't break existing APIs

### For Data Processing (dto, datastructures):
1. **Type safety paramount** - Ensure proper type hints
2. **Edge cases matter** - Handle None, empty, invalid data
3. **Performance sensitive** - DTOs are in hot path
4. **Test with multiple backends** - msgspec, pydantic, etc.

### For Extensions (contrib, plugins):
1. **Isolated changes OK** - Less risk of breaking framework
2. **Follow plugin protocols** - Maintain consistent interfaces
3. **Document thoroughly** - Users need examples
4. **Optional dependencies** - Use lazy imports

### For Infrastructure (middleware, testing):
1. **Consider order dependencies** - Middleware order matters
2. **Test in integration** - Not just unit tests
3. **Handle errors gracefully** - Don't break request pipeline
4. **Provide debugging tools** - Help users troubleshoot

---

## Module Dependency Graph

```
app.py (orchestrator)
   ├── router.py (organizes)
   │   └── handlers/ (processes)
   │       ├── connection/ (abstracts)
   │       ├── dto/ (transforms)
   │       └── response/ (formats)
   ├── middleware/ (filters)
   │   └── connection/ (uses)
   ├── plugins/ (extends)
   │   └── [all modules] (can modify)
   └── config/ (configures)
       └── [all modules] (reads)

contrib/ (integrations)
   ├── plugins/ (implements)
   └── dto/ (extends)

openapi/ (documents)
   └── handlers/ (introspects)

testing/ (tests)
   └── app.py (instantiates)
```

---

## Getting Started with Contributions

### 1. Pick Your Module Level:
- **Beginner**: Start with `exceptions`, `config`, `utils`
- **Intermediate**: Work on `middleware`, `contrib`, `testing`
- **Advanced**: Tackle `handlers`, `dto`, `app.py`

### 2. Understand the Module:
- Read the main module file
- Check tests in `tests/<module_name>/`
- Look for TODOs and FIXMEs
- Review recent PRs for that module

### 3. Common Contribution Patterns:

#### Adding a new middleware:
1. Create in `middleware/your_middleware.py`
2. Inherit from `AbstractMiddleware`
3. Add tests in `tests/unit/test_middleware/`
4. Update documentation

#### Adding a new DTO backend:
1. Create in `dto/your_backend_dto.py`
2. Inherit from `AbstractDTO`
3. Implement required methods
4. Add comprehensive tests

#### Adding a contrib integration:
1. Create directory `contrib/your_library/`
2. Implement plugin protocols
3. Add `__init__.py` with public API
4. Document with examples

### 4. Testing Requirements:
- Unit tests for isolated functionality
- Integration tests for module interactions
- 100% coverage required
- Performance benchmarks for hot paths

### 5. Documentation:
- Docstrings for all public APIs
- Type hints for all parameters
- Examples in docstrings
- Update relevant docs/

---

## Performance Considerations by Module

### Hot Path Modules (optimize aggressively):
- `handlers/` - Called for every request
- `dto/` - Serialization for every response
- `router.py` - Route matching per request
- `middleware/` - Processes all requests

### Cold Path Modules (optimize for clarity):
- `config/` - Only at startup
- `plugins/` - Initialization only
- `openapi/` - Documentation generation
- `cli/` - Command execution

### Memory-Sensitive Modules:
- `dto/` - Caches generated code
- `handlers/` - Stores signatures
- `router.py` - Routing trie structure

---

## Common Pitfalls and Solutions

### 1. Handler Signature Changes:
**Pitfall**: Changing handler signature parsing breaks existing apps
**Solution**: Add new parameters as optional with defaults

### 2. DTO Performance:
**Pitfall**: Generic serialization is slow
**Solution**: Use code generation, cache aggressively

### 3. Middleware Ordering:
**Pitfall**: Middleware depends on execution order
**Solution**: Document dependencies, test combinations

### 4. Type Annotations:
**Pitfall**: Missing or incorrect type hints
**Solution**: Run mypy and pyright, use strict mode

### 5. Async/Sync Mixing:
**Pitfall**: Blocking I/O in async handlers
**Solution**: Use async libraries, run_in_executor for blocking code

---

This guide should help you understand where to contribute based on your expertise level and interests. Each module has its own patterns and conventions, but they all work together to create Litestar's powerful and flexible architecture.