# Front Agent — Follow-Up Workflow (Route A)

User refers to a document, dataset, or resource already in the conversation.

```mermaid
flowchart TD
    A[User message] --> B{Route?}
    B -->|Follow-Up| C

    subgraph front["front_agent"]
        C[Resolve reference from<br>conversation history]
        C --> D{ref marker or<br>dataset URL found?}
        D -->|URL found| E[Call literature_analyse]
        D -->|ID only| F[Call ckan_run package_show]
        D -->|Multiple items| G[Parallel literature_analyse calls]
    end

    E --> DOC
    G --> DOC

    subgraph doc["doc_agent"]
        DOC[Load document]
        DOC --> DOC2[Scan structure + extract passages]
        DOC2 --> DOC3[Return AnalyseResult<br>with highlight URLs]
    end

    DOC3 --> H
    F --> H

    subgraph front2["front_agent"]
        H[Synthesize answer<br>with inline citations]
    end

    H --> Z[Response to user]

    style front fill:#e8e0f0,stroke:#7b5ea7
    style front2 fill:#e8e0f0,stroke:#7b5ea7
    style doc fill:#fff3e0,stroke:#e65100
```

**Forbidden tools:** `find_relevant_groups`, `literature_search`, `package_search`
