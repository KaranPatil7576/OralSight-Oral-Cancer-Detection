# OralSight

OralSight is an educational research prototype for exploring oral image classification. This repository currently contains the frontend and the proposed API contract; it does not include a backend, trained model, dataset, or clinical validation.

> **Important:** This is not a diagnostic tool. Do not use it to diagnose, screen, or guide treatment. The displayed class and model score are research output, not medical advice or a probability of cancer. Consult a qualified dental or medical professional about health concerns.

## Get started

Clone the repository:

```bash
git clone https://github.com/ShreyasRKamble/Oral-Cancer-Detection.git
cd Oral-Cancer-Detection
```

There is no frontend build step or package installation. To preview the static interface, run a local web server from the project directory:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>. The upload and sample interface can be explored, but real predictions require the backend endpoint described below. The frontend posts to the relative path `/api/v1/predictions`, so configure the web server or a development proxy to route that path to the backend. A basic static server alone does not provide this route.

## Project files

- `index.html` — page structure and interface content
- `styles.css` — responsive layout and visual styles
- `app.js` — image selection, validation, API request, and result states
- `API_CONTRACT.md` — proposed request, response, and error contract for backend implementation

## Backend handoff

Implement `POST /api/v1/predictions` as described in [API_CONTRACT.md](API_CONTRACT.md). The frontend sends one image in a `multipart/form-data` field named `image` and accepts JPEG, PNG, or WebP files up to 10 MiB.

A successful JSON response must include:

```json
{
  "prediction_id": "pred_123",
  "predicted_class": "Cancer",
  "score": 0.82,
  "model_version": "v0.1",
  "grad_cam_url": null
}
```

`predicted_class` must be `Cancer` or `Non-Cancer`; `score` must be a number from 0 to 1 and must not be presented as a calibrated cancer probability; `model_version` must be a non-empty string. `grad_cam_url` is optional and may be `null`. Use the error response shape and status codes in the API contract so the frontend can show useful messages and keep the selected image available for retry.

The contract is a proposal. Backend and frontend teammates should agree on it before implementation if the project has selected different infrastructure or behavior.

## Team workflow

1. Agree on the API contract and backend runtime with the team.
2. Implement the prediction service in the backend project or an agreed repository.
3. Route `/api/v1/predictions` from the frontend origin to that service during development and deployment.
4. Keep model files, datasets, credentials, and other large or private artifacts out of this repository. Use the team's approved storage and secret-management process.
5. Test the endpoint with valid images, invalid image data, oversized files, inference failures, and unavailable-model responses before integrating it with users.

## Scope and limitations

This repository contains no prediction implementation. The frontend performs basic file checks for user feedback; the backend must independently validate uploaded content and enforce size limits. The model architecture, preprocessing, training data, and clinical performance are not specified here.
