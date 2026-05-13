
# ToolPermissionPayload


## Properties

Name | Type
------------ | -------------
`name` | string
`description` | string
`permission` | string
`enabled` | boolean
`tags` | Array&lt;string&gt;
`maxCallsPerRun` | number

## Example

```typescript
import type { ToolPermissionPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "name": null,
  "description": null,
  "permission": null,
  "enabled": null,
  "tags": null,
  "maxCallsPerRun": null,
} satisfies ToolPermissionPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ToolPermissionPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


