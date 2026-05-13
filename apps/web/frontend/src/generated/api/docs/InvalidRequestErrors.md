
# InvalidRequestErrors

Check Your Params

## Properties

Name | Type
------------ | -------------
`type` | string
`status` | number
`title` | string
`detail` | [Array&lt;ValidationProblem&gt;](ValidationProblem.md)
`instance` | string

## Example

```typescript
import type { InvalidRequestErrors } from ''

// TODO: Update the object below with actual values
const example = {
  "type": null,
  "status": null,
  "title": null,
  "detail": null,
  "instance": null,
} satisfies InvalidRequestErrors

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as InvalidRequestErrors
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


