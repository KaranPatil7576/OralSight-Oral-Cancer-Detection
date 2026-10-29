# Prediction API contract (proposal)

This contract is the shared interface for the OralSight research prototype. The repository currently contains a static frontend and no backend implementation. Treat this as the agreed proposal for frontend/backend work; confirm it with the backend owner before implementation if the team has already chosen different infrastructure.

## Request

`POST /api/v1/predictions`

- Content type: `multipart/form-data`
- Required field: `image` — one JPEG, PNG, or WebP image
- Maximum image size: 10 MiB
- The browser should send the original file. The server validates file type/content and image size; filename and client MIME type are not trusted.

Example:

```http
POST /api/v1/predictions HTTP/1.1
Content-Type: multipart/form-data; boundary=...
```

Form field: `image=<binary image data>`

## Successful response

`200 OK`, `Content-Type: application/json`

```json
{
  "prediction_id": "pred_123",
  "predicted_class": "Cancer",
  "score": 0.82,
  "model_version": "v0.1",
  "grad_cam_url": null
}
```

- `predicted_class`: exactly `Cancer` or `Non-Cancer` (dataset class, not a diagnosis)
- `score`: number from `0` to `1`, representing the model's score for the returned class; it is not a calibrated cancer probability
- `model_version`: non-empty version string displayed with the result
- `grad_cam_url`: nullable URL for an optional Grad-CAM visualization. When supplied, it should be usable by the browser and tied to this prediction. If omitted or `null`, the UI hides the Grad-CAM section.

## Error response

Errors use `application/json` and this shape:

```json
{
  "error": {
    "code": "UNSUPPORTED_FILE",
    "message": "Upload a JPEG, PNG, or WebP image."
  }
}
```

Supported codes and HTTP statuses:

| Status | Code | Meaning |
| --- | --- | --- |
| `400` | `INVALID_REQUEST` | Missing image or malformed request |
| `413` | `IMAGE_TOO_LARGE` | Image exceeds 10 MiB |
| `415` | `UNSUPPORTED_FILE` | Unsupported format or invalid image data |
| `422` | `PREDICTION_FAILED` | Image accepted but inference could not complete |
| `503` | `MODEL_UNAVAILABLE` | Model service unavailable |

The frontend should show a clear message from `error.message`, retain the selected image for retry, and avoid displaying a prediction on errors. Network failures or unexpected response bodies should be presented as a generic “Prediction service unavailable. Try again.” message.

## Frontend behavior

- Send the request only after the user selects an image and explicitly starts analysis (or when the team chooses the upload-to-infer interaction).
- While pending, show a loading state and prevent duplicate submissions.
- On success, render class, score, and model version. Do not label the score as a probability.
- Render Grad-CAM only when the response provides it.
- Keep the prototype’s educational-use disclaimer visible; this output is not for diagnosis.

## Example error

```json
{
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Image must be 10 MiB or smaller."
  }
}
```
