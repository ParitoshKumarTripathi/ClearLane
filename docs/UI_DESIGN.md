# UI design rationale

The dashboard is intentionally operational rather than decorative.

## Hierarchy

- The product statement appears once at the top.
- Four shift-level metrics summarize the selected operating context in plain dispatch language.
- The map and team dispatch table are the primary work area and share the same width.
- Technical performance and methodology are separated into secondary tabs.

## Controls

- Persistent filters live in the sidebar.
- Labels use sentence case and operational language.
- The team-count control updates the plan immediately.
- Only a few high-value filters are exposed; model parameters remain in configuration.

## Accessibility

- Risk is communicated with text labels, scores, table rows and circle size, not color alone.
- Critical zones and obstruction units are explained briefly on the operations tab.
- The palette uses high-contrast text and restrained status colors.
- Explanations accompany every assigned stop.
- The map has a tabular equivalent for scanning and downloading.

## Responsiveness and performance

- The deployed app reads compressed, precomputed holdout artifacts instead of the 109 MB raw CSV.
- Training is never triggered during a dashboard session.
- The layout uses Streamlit-native responsive columns and persistent sidebar controls.
