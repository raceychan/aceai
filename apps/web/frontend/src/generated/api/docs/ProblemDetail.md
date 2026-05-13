
# ProblemDetail

## Specification:
    - RFC 9457: https://www.rfc-editor.org/rfc/rfc9457.html

This schema provides a standardized way to represent errors in HTTP APIs,
allowing clients to understand error responses in a structured format.

## Properties

Name | Type
------------ | -------------
`type` | string
`status` | number
`title` | string
`detail` | any
`instance` | string

## Example

```typescript
import type { ProblemDetail } from ''

// TODO: Update the object below with actual values
const example = {
  "type": null,
  "status": null,
  "title": null,
  "detail": null,
  "instance": null,
} satisfies ProblemDetail

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ProblemDetail
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


