
# SessionListItemPayload


## Properties

Name | Type
------------ | -------------
`sessionId` | string
`projectId` | string
`projectName` | string
`title` | string
`createdAt` | string
`updatedAt` | string
`eventCount` | number
`totalCostUsd` | number
`threadCount` | number
`activeThread` | [ThreadMetadataPayload](ThreadMetadataPayload.md)
`path` | string

## Example

```typescript
import type { SessionListItemPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "sessionId": null,
  "projectId": null,
  "projectName": null,
  "title": null,
  "createdAt": null,
  "updatedAt": null,
  "eventCount": null,
  "totalCostUsd": null,
  "threadCount": null,
  "activeThread": null,
  "path": null,
} satisfies SessionListItemPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as SessionListItemPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


