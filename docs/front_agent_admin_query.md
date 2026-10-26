# Front Agent — Admin Query (Route D)

User asks about CKAN structure (orgs, groups, users, specific datasets by ID).

```mermaid
flowchart TD
    A[User message] --> B{Route?}
    B -->|Admin Query| C

    subgraph front["front_agent"]
        C[Determine CKAN action<br>e.g. organization_list, group_list,<br>package_show, resource_show]
        C --> D[ckan_run<br>action + parameters]
        D --> E[Format and present results]
    end

    E --> Z[Response to user]

    style front fill:#e8e0f0,stroke:#7b5ea7
```

**Notes:** Single direct `ckan_run` call, no sub-agents involved. Delete/purge actions are blocked.
