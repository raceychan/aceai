
# GuiConfigUpdateRequest


## Properties

Name | Type
------------ | -------------
`provider` | string
`model` | string
`defaultModel` | string
`reasoningLevel` | string
`compressThreshold` | string
`apiTimeoutSeconds` | number
`streamStartTimeoutSeconds` | number
`streamEventTimeoutSeconds` | number
`skillSelectionMode` | string
`enabledSkills` | Array&lt;string&gt;
`disabledProviders` | Array&lt;string&gt;
`apiKey` | string
`toolPermissions` | { [key: string]: string; }
`toolEnabled` | { [key: string]: boolean; }
`toolMaxCalls` | { [key: string]: number; }

## Example

```typescript
import type { GuiConfigUpdateRequest } from ''

// TODO: Update the object below with actual values
const example = {
  "provider": null,
  "model": null,
  "defaultModel": null,
  "reasoningLevel": null,
  "compressThreshold": null,
  "apiTimeoutSeconds": null,
  "streamStartTimeoutSeconds": null,
  "streamEventTimeoutSeconds": null,
  "skillSelectionMode": null,
  "enabledSkills": null,
  "disabledProviders": null,
  "apiKey": null,
  "toolPermissions": null,
  "toolEnabled": null,
  "toolMaxCalls": null,
} satisfies GuiConfigUpdateRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as GuiConfigUpdateRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


