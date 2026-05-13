
# ProviderOptionPayload


## Properties

Name | Type
------------ | -------------
`label` | string
`value` | string
`authMode` | string
`apiKeyEnv` | string

## Example

```typescript
import type { ProviderOptionPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "label": null,
  "value": null,
  "authMode": null,
  "apiKeyEnv": null,
} satisfies ProviderOptionPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ProviderOptionPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


