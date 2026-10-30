# Validation record

Checks performed on 29 October 2026, with Python 3.12 on Linux. Only inference and software checks were run; no training occurred.

## Passed

- Installed the final pinned Python requirements in an isolated environment; `pip check` reported no broken requirements.
- Loaded the actual saved Keras model using the TensorFlow CPU backend: input 160 × 160 × 3, output 25 classes, total 1,016,201 parameters.
- Verified finite softmax scores and successful no-hand processing on a blank frame using MediaPipe 1.1.1.
- Ran both automatic cropping and fixed guide-box inference on the A reference image from the existing project contact sheet. Both predicted A. This is a smoke test on an existing illustration, not an independent accuracy result.
- Processed six actual PyAV video frames through the processor and verified cleanup on stream end.
- Ran four decision-rule regression tests: ambiguity rejection, score-gap rejection, stale-letter removal, and independent/time-limited vote history.
- Used Streamlit AppTest to check page initialization, photo mode, actual photo inference/result rendering, adding a labelled trial, CSV rendering, guide-box controls and input flip controls. The WebRTC transport was stubbed because AppTest does not provide a real browser session. Model inference and photo processing were real.
- Compiled the Python sources and checked the ZIP contents.
- Verified that the model SHA-256 exactly matches the working desktop package:
  `21eff4561c56f01a7be0c222b852d11427389f8f2257eaf87b9b35f51b89b657`.

## Deployment fixes included

MediaPipe 1.1.1 avoids the old JAX/TensorFlow dependency conflict. WebRTC libraries load before TensorFlow to avoid a native-library initialization conflict observed during tests. PyArrow is pinned to a version compatible with NumPy 1.26. Required Linux graphics libraries are listed in `packages.txt`; the local test supplied libGLESv2 from a separately extracted Ubuntu package.

## Still to verify

- A real HTTPS deployment on the user's Streamlit account.
- Browser camera selection, permissions, WebRTC negotiation, mobile behavior and TURN/network requirements.
- Visual layout in a real browser. AppTest checks application elements but does not render a browser screenshot.
- End-to-end webcam latency, multiple simultaneous visitors and cloud memory limits.
- Real-world accuracy on unseen people/sessions, especially A/K/V. No webcam samples were supplied with this request.
- Numerical equivalence between the desktop PyTorch backend and this cloud TensorFlow backend has not been measured.

The classifier weights were not improved or retrained. The changes improve framing visibility, handle ambiguous/old results more carefully, and provide a practical way to measure remaining errors. They do not establish a higher recognition accuracy.
