
# IdeaItemPayload


## Properties

Name | Type
------------ | -------------
`index` | number
`ideaId` | string
`createdAt` | string
`projectId` | string
`projectName` | string
`workspace` | string
`content` | string
`sourceSessionId` | string

## Example

```typescript
import type { IdeaItemPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "index": null,
  "ideaId": null,
  "createdAt": null,
  "projectId": null,
  "projectName": null,
  "workspace": null,
  "content": null,
  "sourceSessionId": null,
} satisfies IdeaItemPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as IdeaItemPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


