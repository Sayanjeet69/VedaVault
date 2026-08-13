# Source Manifest Schema

`Data/sources.yaml` is VedaVault's canonical registry for textual sources. Each
item in `sources` describes one source that may later be ingested. Keep source
content in `Data/Raw/`; the manifest records its expected location but does not
contain the text itself.

## Structure

The manifest has a numeric `version` and a `sources` list. Every source entry
must include every field below. Use quoted strings for values containing a colon
or other YAML-significant characters.

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable, unique, lowercase hyphenated identifier for the source. |
| `title` | string | Human-readable title of the work or collection. |
| `tradition` | string | Religious, philosophical, or literary tradition represented. |
| `language` | string | Primary language of the source text. |
| `source_url` | string | Canonical URL from which the source is obtained. Use a clearly marked `TODO` until verified. |
| `license` | string | License or rights statement for the exact source edition. Use a clearly marked `TODO` until verified. |
| `content_path` | string | Repository-relative path to the source content beneath `Data/Raw/`. Use a clearly marked `TODO` until content is acquired. |
| `format` | string | File representation expected at `content_path`, for example `text`, `markdown`, `html`, `json`, or `pdf`. |
| `status` | string | Lifecycle state. Use `example` for documentation-only entries; only a future ingestion workflow should process approved states. |

## Example

The initial Bhagavad Gita record is intentionally an `example`. Its URL,
license, and local content path remain `TODO` until a source edition and its
usage rights have been verified. It must not be treated as downloadable data.

## Validation

Run the dependency-free validator from the repository root:

```powershell
python Scripts/validate_sources.py
```

The validator checks manifest structure and required, non-empty fields. It
supports the constrained YAML shape used by this manifest: a top-level
`version`, a `sources` list, and indented scalar mappings for each source.
