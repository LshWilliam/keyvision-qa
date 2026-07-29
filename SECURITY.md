# Security policy

## Reporting

Please open a private GitHub security advisory for suspected vulnerabilities. Do not attach
credentials, proprietary images, or personal data to a public issue.

## Data and model safety

- Treat checkpoints, ONNX files, NumPy archives, and annotations as untrusted input.
- Load only files from known sources; PyTorch checkpoints can contain unsafe serialized objects.
- Keep credentials in an external secret manager or untracked environment variables.
- Do not expose the Gradio development server directly to an untrusted network without
  authentication, TLS termination, upload limits, and patch management.
- Validate image size and format at a production boundary to mitigate decompression bombs and
  resource exhaustion.

Only the latest minor release receives security fixes while this project is pre-1.0.

