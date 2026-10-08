# TSE analytical model DAG

```mermaid
flowchart LR
    RAW["RAW<br/>immutable TSE objects"]
    B["Bronze<br/>source interpretation"]
    S["Silver<br/>canonical grain"]
    P["Physical<br/>implementation helper"]
    G["Gold<br/>facts / dimensions / reconciliation"]
    M["Semantic<br/>consumer metrics"]

    RAW --> B
    B --> S
    S --> P
    B --> G
    S --> G
    P --> G
    G --> M
```

Allowed dependency direction:

```text
RAW -> Bronze -> Silver -> Gold -> Semantic
                 |
                 +-> Physical -> Gold
```

Backward dependencies are architectural violations and should fail CI.
