# Research Agent — Follow-Up Workflow (Route A)

User refers to a document, dataset, or resource already in the conversation.
Skips the 5-phase research process entirely.

```mermaid
flowchart TD
    A[User message] --> B{Route?}
    B -->|Follow-Up| C

    subgraph research["research_agent"]
        C[Resolve reference from<br>conversation history]
        C --> D{ref marker or<br>dataset URL found?}
        D -->|URL found| E[Call literature_analyse]
        D -->|Multiple items| F[Parallel literature_analyse calls]
        D -->|ID only| G[Call ckan_run package_show]
    end

    E --> DOC
    F --> DOC

    subgraph doc["doc_agent"]
        DOC[Load document]
        DOC --> DOC2[Scan structure + extract passages]
        DOC2 --> DOC3[Return AnalyseResult<br>with highlight URLs]
    end

    DOC3 --> H
    G --> H

    subgraph research2["research_agent"]
        H[Synthesize answer<br>with inline citations]
    end

    H --> Z[Response to user]

    style research fill:#e0e8f0,stroke:#1565c0
    style research2 fill:#e0e8f0,stroke:#1565c0
    style doc fill:#fff3e0,stroke:#e65100
```

**Forbidden tools:** `find_relevant_groups`, `literature_search`, `ckan_explore`
