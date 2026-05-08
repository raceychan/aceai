# Provider authentication

AceAI records each provider's authentication mode in
`aceai/agent/provider_catalog.yaml`.

## API key providers

API key providers charge through the provider API account. AceAI shows an
`api_key` field in setup and settings, and reads the provider-specific
environment variable when the saved config does not already contain a key.

Current API key providers:

- `openai`: uses `OPENAI_API_KEY`.
- `deepseek`: uses `DEEPSEEK_API_KEY`.

## Subscription providers

Subscription providers use an existing local subscription login instead of a
manual API key. AceAI does not show an `api_key` field for these providers in
setup or settings. The config still stores an internal credential sentinel so
the provider adapter can resolve the local login at runtime.

Current subscription providers:

- `codex`: shown as `Codex (subscription)`. It uses the Codex CLI login in
  `~/.codex/auth.json`. AceAI stores
  the internal sentinel `codex-cli` and the provider resolves the access token
  from the local Codex auth file when a run starts.
