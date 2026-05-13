
# ThreadMetadataPayload


## Properties

Name | Type
------------ | -------------
`sessionId` | string
`threadId` | string
`role` | string
`title` | string
`status` | string
`agentId` | string
`parentThreadId` | string
`parentRunId` | string
`parentToolCallId` | string
`metadata` | object
`createdAt` | string
`updatedAt` | string

## Example

```typescript
import type { ThreadMetadataPayload } from ''

// TODO: Update the object below with actual values
const example = {
  "sessionId": null,
  "threadId": null,
  "role": null,
  "title": null,
  "status": null,
  "agentId": null,
  "parentThreadId": null,
  "parentRunId": null,
  "parentToolCallId": null,
  "metadata": null,
  "createdAt": null,
  "updatedAt": null,
} satisfies ThreadMetadataPayload

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ThreadMetadataPayload
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


