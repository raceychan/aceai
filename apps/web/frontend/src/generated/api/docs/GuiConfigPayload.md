
# GuiConfigPayload


## Properties

Name | Type
------------ | -------------
`projectName` | string
`gitBranch` | string
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
`apiKeySet` | boolean
`apiKeyEnv` | string
`configPath` | string
`providers` | [Array&lt;ProviderOptionPayload&gt;](ProviderOptionPayload.md)
`models` | [Array&lt;ModelOptionPayload&gt;](ModelOptionPayload.md)
`modelsByProvider` | { [key: string]: [Array&lt;ModelOptionPayload&gt;](ModelOptionPayload.md); }
`reasoningOptions` | Array&lt;string&gt;
`skills` | [Array&lt;SkillItemPayload&gt;](SkillItemPayload.md)
`tools` | [Array&lt;ToolPermissionPayload&gt;](ToolPermissionPayload.md)

## Example

```typescript
import type { GuiConfigPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "projectName": null,
  "gitBranch": null,
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
  "apiKeySet": null,
  "apiKeyEnv": null,
  "configPath": null,
  "providers": null,
  "models": null,
  "modelsByProvider": null,
  "reasoningOptions": null,
  "skills": null,
  "tools": null,
} satisfies GuiConfigPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as GuiConfigPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


