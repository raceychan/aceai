
# SkillItemPayload


## Properties

Name | Type
------------ | -------------
`name` | string
`description` | string
`location` | string
`source` | string
`builtin` | boolean
`enabled` | boolean

## Example

```typescript
import type { SkillItemPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "name": null,
  "description": null,
  "location": null,
  "source": null,
  "builtin": null,
  "enabled": null,
} satisfies SkillItemPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as SkillItemPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


