# Litestar Architecture Diagrams (Detailed & Descriptive)

## 1. High-Level Architecture Overview with Detailed Descriptions

```mermaid
flowchart TB
    subgraph AppLayer["🎯 Application Layer - User-Facing Components"]
        App["Litestar App<br/>────────────<br/>• Main entry point for your application<br/>• Manages configuration & lifecycle<br/>• Registers routes, middleware, plugins<br/>• Similar to Flask() or FastAPI()"]
        
        Router["Router<br/>────────────<br/>• Groups related endpoints together<br/>• Defines URL path prefixes<br/>• Can have own middleware/guards<br/>• Think: /api/v1/* routes"]
        
        Controller["Controller<br/>────────────<br/>• Class-based route organization<br/>• Groups related business logic<br/>• Shared dependencies & guards<br/>• Like Django ViewSets"]
    end
    
    subgraph ASGILayer["⚡ ASGI Layer - Protocol Implementation"]
        ASGIRouter["ASGI Router<br/>────────────<br/>• Converts HTTP to ASGI events<br/>• Main request dispatcher<br/>• Handles async communication<br/>• Core of async performance"]
        
        RoutingTrie["Routing Trie<br/>────────────<br/>• Tree structure for URL matching<br/>• O(n) path lookup complexity<br/>• Handles path parameters<br/>• /users/{id}/posts/{pid}"]
        
        ASGIApp["ASGI Applications<br/>────────────<br/>• Standard Python async apps<br/>• Handle scope, receive, send<br/>• WebSocket & HTTP support<br/>• Interoperable with other frameworks"]
    end
    
    subgraph HandlerLayer["📝 Handler Layer - Request Processing"]
        HTTPHandler["HTTP Handlers<br/>────────────<br/>• @get, @post, @put, @delete<br/>• Handle REST API requests<br/>• Return JSON, HTML, files<br/>• Sync or async functions"]
        
        WSHandler["WebSocket Handlers<br/>────────────<br/>• Real-time bidirectional comm<br/>• @websocket decorator<br/>• Streaming & pub/sub support<br/>• Chat, notifications, live data"]
        
        ASGIHandler["ASGI Handlers<br/>────────────<br/>• Raw ASGI interface access<br/>• Maximum control & flexibility<br/>• Custom protocols<br/>• Advanced use cases"]
    end
    
    subgraph MiddlewareLayer["🛡️ Middleware Layer - Request/Response Pipeline"]
        AuthMiddleware["Authentication<br/>────────────<br/>• JWT, Session, OAuth<br/>• User identification<br/>• Token validation<br/>• Sets request.user"]
        
        CORSMiddleware["CORS<br/>────────────<br/>• Cross-Origin Resource Sharing<br/>• Browser security policy<br/>• Allows frontend API calls<br/>• Configurable origins"]
        
        CompressionMiddleware["Compression<br/>────────────<br/>• Gzip, Brotli compression<br/>• Reduces bandwidth usage<br/>• Automatic for responses<br/>• 60-80% size reduction"]
        
        CSRFMiddleware["CSRF Protection<br/>────────────<br/>• Prevents forged requests<br/>• Token-based validation<br/>• Form & API protection<br/>• Security best practice"]
        
        RateLimitMiddleware["Rate Limiting<br/>────────────<br/>• Request throttling<br/>• DDoS protection<br/>• Per-user/IP limits<br/>• Redis/memory backed"]
    end
    
    subgraph DataLayer["📦 Data Layer - Serialization & Validation"]
        DTO["DTOs - Data Transfer Objects<br/>────────────<br/>• Define API contracts<br/>• Field inclusion/exclusion<br/>• Rename & transform fields<br/>• Security by design"]
        
        Serialization["Serialization Engine<br/>────────────<br/>• JSON/MessagePack conversion<br/>• 10-50x faster with msgspec<br/>• Type-safe transformations<br/>• Handles nested objects"]
        
        Validation["Validation System<br/>────────────<br/>• Type checking at runtime<br/>• Constraint validation<br/>• Custom validators<br/>• Detailed error messages"]
    end
    
    subgraph PluginSystem["🔌 Plugin System - Extensibility"]
        InitPlugin["Init Plugins<br/>────────────<br/>• Modify app at startup<br/>• Add global config<br/>• Register components<br/>• Database connections"]
        
        SerializationPlugin["Serialization Plugins<br/>────────────<br/>• Custom type support<br/>• Pydantic, attrs integration<br/>• Custom encoders/decoders<br/>• Domain objects"]
        
        OpenAPIPlugin["OpenAPI Plugins<br/>────────────<br/>• API documentation<br/>• Swagger/ReDoc UI<br/>• Schema customization<br/>• Client code generation"]
        
        CLIPlugin["CLI Plugins<br/>────────────<br/>• Custom commands<br/>• Management tasks<br/>• Database migrations<br/>• Development tools"]
    end
    
    subgraph SupportingSystems["🔧 Supporting Systems - Core Services"]
        DI["Dependency Injection<br/>────────────<br/>• Automatic wiring<br/>• Request/app scoped<br/>• Testing friendly<br/>• Clean architecture"]
        
        Events["Event System<br/>────────────<br/>• Lifecycle hooks<br/>• Before/after request<br/>• Custom events<br/>• Decoupled logic"]
        
        Stores["Storage System<br/>────────────<br/>• Session management<br/>• Cache storage<br/>• File uploads<br/>• State persistence"]
        
        Templates["Template Engine<br/>────────────<br/>• Jinja2 integration<br/>• HTML rendering<br/>• Email templates<br/>• Server-side rendering"]
        
        Channels["Channels/PubSub<br/>────────────<br/>• Message broadcasting<br/>• WebSocket rooms<br/>• Event distribution<br/>• Real-time updates"]
    end
    
    App --> Router
    Router --> Controller
    Controller --> HTTPHandler
    Controller --> WSHandler
    
    App --> ASGIRouter
    ASGIRouter --> RoutingTrie
    ASGIRouter --> ASGIApp
    
    ASGIApp --> AuthMiddleware
    AuthMiddleware --> CORSMiddleware
    CORSMiddleware --> HTTPHandler
    
    HTTPHandler --> DTO
    DTO --> Serialization
    
    App --> InitPlugin
    App --> DI
    
    DI --> HTTPHandler
    Events --> App
```

## 2. Request Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant Server as ASGI Server
    participant Router as ASGIRouter
    participant Trie as RoutingTrie
    participant MW as Middleware
    participant Handler as RouteHandler
    participant DI as DI System
    participant DTO
    participant Response
    
    Client->>Server: HTTP Request
    Server->>Router: scope, receive, send
    
    Router->>Router: Normalize path
    Router->>Trie: Parse path to route
    Trie->>Trie: Match route pattern
    Trie-->>Router: Route, Handler, Params
    
    Router->>MW: Process request
    
    loop Middleware Stack
        MW->>MW: Before request hooks
    end
    
    MW->>Handler: Call handler
    Handler->>DI: Resolve dependencies
    DI-->>Handler: Injected dependencies
    
    Handler->>DTO: Validate/Parse request data
    DTO-->>Handler: Validated data
    
    Handler->>Handler: Execute business logic
    Handler->>Response: Create response
    
    Response->>DTO: Serialize response data
    DTO-->>Response: Serialized data
    
    Response->>MW: Process response
    
    loop Middleware Stack reverse
        MW->>MW: After response hooks
    end
    
    MW->>Server: Send response
    Server->>Client: HTTP Response
```

## 3. Core Module Structure

```mermaid
flowchart LR
    subgraph CoreModules["Core Modules"]
        app["app.py - Main Application"]
        router["router.py - Route Management"]
        controller["controller.py - Class Controllers"]
    end
    
    subgraph ConnectionLayer["Connection Layer"]
        request["connection/request.py - HTTP Request"]
        websocket["connection/websocket.py - WebSocket"]
        base_conn["connection/base.py - Base Connection"]
    end
    
    subgraph HandlerTypes["Handler Types"]
        http_handler["handlers/http_handlers/ - GET POST etc"]
        ws_handler["handlers/websocket_handlers/ - WebSocket handlers"]
        asgi_handler["handlers/asgi_handlers.py - Raw ASGI"]
    end
    
    subgraph ASGIImplementation["ASGI Implementation"]
        asgi_router["_asgi/asgi_router.py - ASGI Router"]
        routing_trie["_asgi/routing_trie/ - Path Matching"]
    end
    
    subgraph DataProcessing["Data Processing"]
        dto_base["dto/base_dto.py - Abstract DTO"]
        msgspec_dto["dto/msgspec_dto.py - Msgspec DTO"]
        dataclass_dto["dto/dataclass_dto.py - Dataclass DTO"]
    end
    
    app --> router
    router --> controller
    controller --> http_handler
    controller --> ws_handler
    
    app --> asgi_router
    asgi_router --> routing_trie
    
    http_handler --> request
    ws_handler --> websocket
    request --> base_conn
    websocket --> base_conn
    
    http_handler --> dto_base
    dto_base --> msgspec_dto
    dto_base --> dataclass_dto
```

## 4. Routing Trie Data Structure

```mermaid
flowchart TD
    Root["Root Node /"]
    
    Root --> api["/api"]
    Root --> static["/static"]
    Root --> health["/health"]
    
    api --> v1["/v1"]
    api --> v2["/v2"]
    
    v1 --> users_v1["/users"]
    v1 --> posts_v1["/posts"]
    
    v2 --> users_v2["/users"]
    v2 --> products["/products"]
    
    users_v1 --> user_id["/{user_id} - Parameter Node"]
    posts_v1 --> post_id["/{post_id} - Parameter Node"]
    
    user_id --> profile["/profile"]
    user_id --> settings["/settings"]
    
    style user_id fill:#ffcccc
    style post_id fill:#ffcccc
```

## 5. Dependency Injection System

```mermaid
flowchart TB
    subgraph DIResolution["DI Resolution"]
        Request[Request Scope]
        Handler[Handler Definition]
        Dependencies[Dependencies Dict]
        Provider[Provide Function]
    end
    
    subgraph DependencyScopes["Dependency Scopes"]
        AppScope["App Scope - Singleton"]
        RequestScope["Request Scope - Per Request"]
        WebsocketScope["WebSocket Scope - Per Connection"]
    end
    
    subgraph ResolutionProcess["Resolution Process"]
        Parse[Parse Signature]
        Check[Check Cache]
        Resolve[Resolve Dependencies]
        Inject[Inject Values]
        Cache[Cache Result]
    end
    
    Handler --> Parse
    Parse --> Check
    Check -->|Hit| Inject
    Check -->|Miss| Resolve
    Resolve --> Dependencies
    Dependencies --> Provider
    Provider --> AppScope
    Provider --> RequestScope
    Provider --> WebsocketScope
    Resolve --> Cache
    Cache --> Inject
```

## 6. Plugin Architecture

```mermaid
classDiagram
    class PluginProtocol {
        <<interface>>
    }
    
    class InitPlugin {
        <<interface>>
        +on_app_init(AppConfig) AppConfig
    }
    
    class SerializationPlugin {
        <<interface>>
        +supports_type(type) bool
        +serialize(value) Any
        +deserialize(value, type) Any
    }
    
    class OpenAPISchemaPlugin {
        <<interface>>
        +is_plugin_supported_field(field) bool
        +to_openapi_schema(field) Schema
    }
    
    class CLIPlugin {
        <<interface>>
        +on_cli_init(Group) None
    }
    
    class DIPlugin {
        <<interface>>
        +has_typed_init(type) bool
        +get_typed_init(type) callable
    }
    
    PluginProtocol <|-- InitPlugin
    PluginProtocol <|-- SerializationPlugin
    PluginProtocol <|-- OpenAPISchemaPlugin
    PluginProtocol <|-- CLIPlugin
    PluginProtocol <|-- DIPlugin
    
    class PydanticPlugin {
        +on_app_init(AppConfig) AppConfig
        +supports_type(type) bool
        +to_openapi_schema(field) Schema
    }
    
    class SQLAlchemyPlugin {
        +on_app_init(AppConfig) AppConfig
        +has_typed_init(type) bool
    }
    
    class PrometheusPlugin {
        +on_app_init(AppConfig) AppConfig
        +on_cli_init(Group) None
    }
    
    InitPlugin <|.. PydanticPlugin
    SerializationPlugin <|.. PydanticPlugin
    OpenAPISchemaPlugin <|.. PydanticPlugin
    
    InitPlugin <|.. SQLAlchemyPlugin
    DIPlugin <|.. SQLAlchemyPlugin
    
    InitPlugin <|.. PrometheusPlugin
    CLIPlugin <|.. PrometheusPlugin
```

## 7. Middleware Stack Execution

```mermaid
flowchart TB
    Request[Incoming Request]
    
    CORS[CORS Middleware]
    Auth[Authentication Middleware]
    RateLimit[Rate Limit Middleware]
    Compression[Compression Middleware]
    Custom[Custom Middleware]
    
    Handler[Route Handler]
    
    Response[Outgoing Response]
    
    Request --> CORS
    CORS --> Auth
    Auth --> RateLimit
    RateLimit --> Compression
    Compression --> Custom
    Custom --> Handler
    
    Handler --> Custom
    Custom --> Compression
    Compression --> RateLimit
    RateLimit --> Auth
    Auth --> CORS
    CORS --> Response
    
    style Handler fill:#90EE90
```

## 8. DTO and Serialization Layer

```mermaid
flowchart LR
    subgraph DTOSystem["DTO System"]
        AbstractDTO["Abstract DTO - Base Class"]
        
        MsgspecDTO["Msgspec DTO - Default/Fast"]
        DataclassDTO[Dataclass DTO]
        PydanticDTO["Pydantic DTO - Via Plugin"]
        
        DTOConfig["DTO Config - exclude rename etc"]
        DTOData["DTO Data - Parsed Container"]
    end
    
    subgraph SerializationFlow["Serialization Flow"]
        RawData[Raw Request Data]
        TypeHints[Type Hints]
        Validation[Validation]
        Transform[Transformation]
        SerializedData[Serialized Response]
    end
    
    subgraph CodeGeneration["Code Generation"]
        CodeGen["Code Generator codegen_backend.py"]
        ParsedModel[Parsed Model]
        GeneratedCode[Generated Functions]
    end
    
    AbstractDTO --> MsgspecDTO
    AbstractDTO --> DataclassDTO
    AbstractDTO --> PydanticDTO
    
    AbstractDTO --> DTOConfig
    DTOConfig --> DTOData
    
    RawData --> TypeHints
    TypeHints --> Validation
    Validation --> Transform
    Transform --> DTOData
    DTOData --> SerializedData
    
    AbstractDTO --> CodeGen
    CodeGen --> ParsedModel
    ParsedModel --> GeneratedCode
    GeneratedCode --> Transform
```

## 9. Event System State Machine

```mermaid
stateDiagram-v2
    [*] --> AppStartup
    
    AppStartup --> OnStartup: Execute startup hooks
    OnStartup --> AppInitialized: Plugins initialized
    
    AppInitialized --> RequestReceived
    
    RequestReceived --> BeforeRequest: Execute before_request hooks
    BeforeRequest --> RouteMatched: Find route handler
    
    RouteMatched --> HandlerExecuting: Resolve dependencies
    HandlerExecuting --> HandlerExecuted: Business logic
    
    HandlerExecuted --> AfterRequest: Execute after_request hooks
    AfterRequest --> ResponsePrepared
    
    ResponsePrepared --> BeforeSend: Execute before_send hooks
    BeforeSend --> ResponseSent
    
    ResponseSent --> AfterResponse: Execute after_response hooks
    AfterResponse --> RequestComplete
    
    RequestComplete --> RequestReceived: Next request
    RequestComplete --> AppShutdown: Server stopping
    
    AppShutdown --> OnShutdown: Execute shutdown hooks
    OnShutdown --> [*]
    
    HandlerExecuting --> ExceptionOccurred: Error
    ExceptionOccurred --> AfterException: Execute after_exception hooks
    AfterException --> ErrorResponse
    ErrorResponse --> ResponseSent
```

## 10. Simplified Complete Architecture - Beginner Friendly

```mermaid
flowchart TB
    Client["Client - Browser/Mobile/API<br/>────────────<br/>• Makes HTTP/WebSocket requests<br/>• Sends JSON/Form data<br/>• Receives responses"]
    
    subgraph Litestar["🚀 Litestar Application"]
        App["App Core<br/>────────────<br/>• The main Litestar() instance<br/>• Configures entire application<br/>• Entry point for all requests"]
        
        subgraph Routing["🗺️ Routing - URL to Code Mapping"]
            Router["Router<br/>────────────<br/>• Maps URLs to functions<br/>• /users → get_users()<br/>• Pattern matching"]
            Trie["Routing Trie<br/>────────────<br/>• Fast URL lookup tree<br/>• Handles /users/{id}<br/>• Extracts parameters"]
        end
        
        subgraph Handlers["👨‍💻 Handlers - Your Business Logic"]
            HTTP["HTTP Handler<br/>────────────<br/>• Your API functions<br/>• @get, @post decorators<br/>• Return JSON/HTML"]
            WS["WebSocket Handler<br/>────────────<br/>• Real-time connections<br/>• Chat, notifications<br/>• Two-way communication"]
            ASGI["ASGI Handler<br/>────────────<br/>• Advanced usage<br/>• Custom protocols<br/>• Full control"]
        end
        
        subgraph Middleware["🛡️ Middleware - Request Filters"]
            CORS["CORS<br/>────────────<br/>• Allow browser requests<br/>• Cross-origin control<br/>• Frontend compatibility"]
            Auth["Auth<br/>────────────<br/>• User authentication<br/>• JWT/Session tokens<br/>• Access control"]
            Cache["Cache<br/>────────────<br/>• Store responses<br/>• Improve performance<br/>• Reduce DB hits"]
        end
        
        subgraph Data["📊 Data Layer - Input/Output"]
            DTO["DTO<br/>────────────<br/>• API contracts<br/>• Hide sensitive fields<br/>• Transform data shape"]
            Validation["Validation<br/>────────────<br/>• Check input data<br/>• Type verification<br/>• Business rules"]
            Serialization["Serialization<br/>────────────<br/>• Python ↔ JSON<br/>• Fast conversion<br/>• Nested objects"]
        end
        
        subgraph DI["💉 DI System - Smart Wiring"]
            Container["DI Container<br/>────────────<br/>• Manages dependencies<br/>• Auto-injection<br/>• Database, services"]
            Scopes["Scopes<br/>────────────<br/>• Request: per-request<br/>• App: singleton<br/>• Lifecycle management"]
        end
        
        subgraph Plugins["🔧 Plugins - Extensions"]
            PydanticP["Pydantic<br/>────────────<br/>• Data validation<br/>• Popular schemas<br/>• Type safety"]
            SQLAP["SQLAlchemy<br/>────────────<br/>• Database ORM<br/>• SQL queries<br/>• Migrations"]
            OpenAPISpec["OpenAPI<br/>────────────<br/>• Auto documentation<br/>• Swagger UI<br/>• API testing"]
        end
    end
    
    Client -->|"1. HTTP Request"| App
    App -->|"2. Find route"| Router
    Router -->|"3. Match pattern"| Trie
    Trie -->|"4. Apply filters"| Middleware
    Middleware -->|"5. Call handler"| HTTP
    Middleware -->|"5. Or WebSocket"| WS
    HTTP -->|"6. Get dependencies"| Container
    Container -->|"7. Inject services"| Scopes
    HTTP -->|"8. Process data"| DTO
    DTO -->|"9. Validate input"| Validation
    Validation -->|"10. Convert to JSON"| Serialization
    App -->|"Uses"| PydanticP
    App -->|"Uses"| SQLAP
    App -->|"Uses"| OpenAPISpec
    
    style Client fill:#e1f5fe
    style App fill:#90ee90
    style HTTP fill:#ffffcc
    style DTO fill:#ffcccc
```