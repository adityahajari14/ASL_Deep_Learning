# ASL alphabet recognition — cloud app

A simple Streamlit interface for the existing college-project model. It supports a browser webcam, photo testing, a model-input preview and a downloadable test log. No training dataset is needed to run it.

## Deploy on Streamlit Community Cloud

1. Unzip this package. Create a GitHub repository and upload the **contents of ASL_Cloud** to its root. `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml` and `model_results/` must keep their relative paths. Include the bundled `hand_landmarker.task` if present; otherwise it downloads on first launch. Do not upload a ZIP instead of the files.
2. Sign in at https://share.streamlit.io/ using your GitHub account. Create an app, choose the repository and branch, and set the main file to `app.py`.
3. In Advanced settings select **Python 3.12**. This cloud environment is separate from your laptop's Python 3.14; you do not need to replace your laptop Python.
4. Deploy, wait for installation, then open the HTTPS app link. The first start loads the model. Use the video controls to choose a camera, press START and allow camera access.
5. Test both Live camera and Photo test from a second device. Check A, K, V and several other letters before sharing the demo.

A GitHub/Streamlit account and deployment action are still required. This package is not itself a published URL. Free hosting runs on shared CPU resources and can sleep; it does not guarantee a particular frame rate or capacity.

### Enable the live camera (required on Streamlit Cloud)

Streamlit Community Cloud blocks direct WebRTC connections, so the camera will not start after pressing START unless a TURN relay is configured. The app shows a warning when none is set.

Recommended option: Cloudflare TURN (free tier).

1. Sign in at https://dash.cloudflare.com and open **Realtime → TURN Server** (formerly "Calls").
2. Create a TURN key. Copy the **Turn Token ID** and the **API Token** (shown only once).
3. In Streamlit → your app → **Settings → Secrets** add:

```toml
CLOUDFLARE_TURN_KEY_ID = "your-turn-token-id"
CLOUDFLARE_TURN_API_TOKEN = "your-api-token"
```

Or, if you have another TURN provider (for example Metered), use its static credentials. Separate several URLs with commas:

```toml
TURN_URL = "turn:your-relay-host:80,turns:your-relay-host:443"
TURN_USERNAME = "your-username"
TURN_PASSWORD = "your-password"
```

Save, then reboot the app. Never commit these values to GitHub. Photo test works without a relay.

## Run locally for inference tests (optional)

Use Python 3.12 in a separate virtual environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests
streamlit run app.py
```

On Linux, install the system packages listed in `packages.txt`. MediaPipe depends on OpenCV contrib; this environment installs only that OpenCV variant. Do not add another headless/non-headless OpenCV wheel on top of it.

## How to use it

- Start with automatic framing, coverage 1.8, score threshold 0.75 and score gap 0.15.
- Keep one whole hand and some wrist visible. Use a plain background and even lighting. Avoid very small or partly cropped hands.
- Look at **Exact image seen by the CNN**. Adjust coverage if too much or too little background is included. Settings affect recognition immediately.
- If automatic detection misses a closed fist, use **Fixed guide box** and position your entire hand and wrist inside the square. This intentionally bypasses the hand detector; it can also classify an empty box incorrectly.
- Compare the reference examples before changing the input flip. Flipping alters recognition input; it is not just a mirrored display. Keep one setting for a formal test.
- A blank recognized letter means the app is uncertain, sees no sign, or is waiting for enough agreement. Model scores are not guarantees of correctness.
- Photo test performs one classification without temporal voting. Upload images or take a camera snapshot; the phone's EXIF rotation is respected. Images larger than 1280 pixels are reduced before hand detection.

## What changed for real-world use

The trained weights are unchanged. The app reuses the wider context crop from the corrected desktop version, preserves RGB and raw 0–255 values, and leaves normalization inside the model. It does not assume that a crop change alone fixes generalization.

New safeguards include rejection when the two leading labels are too close, removal of stale voted letters during sign transitions, a timeout on voting history, and guidance when a detected hand is tiny or near the frame edge. A fixed-box fallback can distinguish a hand-detector failure from a CNN classification failure. Each visitor has a separate hand tracker and vote history; the model weights are shared under a lock.

Thresholds and image-size checks are heuristics, not calibrated settings. They can reduce wrong displayed predictions by rejecting more frames, but that is not necessarily an increase in classifier accuracy. No handcrafted A/K/V label overrides or automatic flips are used.

## Measure the actual issue

1. Collect consented photos from several people and multiple sessions, with varied backgrounds and lighting. Include every supported class and no-sign scenes. Do not use training images as webcam evidence.
2. Use one collection for choosing framing/settings. Reserve different people or sessions for final evaluation. Avoid testing on near-identical frames used for tuning.
3. In Photo test, enter the actual sign and add each trial to the log once. Download the CSV. Log photos with missed detections too: they count as failures in the displayed top-1 accuracy. Review the status column to separate detection failures from CNN mistakes.
4. Report raw top-1 accuracy, accepted-prediction accuracy, rejection/detection-failure rate and per-class confusion. Test live sign transitions separately; still-image results do not measure temporal voting.
5. If correctly framed A/K/V examples still fail repeatedly, download labelled test crops. Create a separate consented webcam training/validation collection, fine-tune on **Kaggle**, and keep the final test people/sessions unseen. Do not train on this final evaluation set. No retraining is performed by this app or this package.

The original dataset test accuracy was 99.02%; real-world webcam accuracy remains unmeasured. Dataset backgrounds, hand appearance, handedness, viewing angle and lighting can differ substantially from a webcam.

## Files to explain to your teacher

- `app.py`: page, controls, camera/photo input, result display and test-log download.
- `model.py`: loads the existing CNN, resizes RGB crops and returns 25 scores. Cloud uses TensorFlow CPU through Keras; the desktop version used PyTorch through Keras.
- `hand_detector.py`: MediaPipe hand detection and square wrist/context crop.
- `recognition.py`: selects a crop, calls the model and applies decision rules.
- `decision.py`: uncertainty checks and five-of-seven voting.
- `video.py`: browser-frame processing and per-visitor state.
- `model_results/`: saved weights, label/config file and reference images.
- `smoke_test.py`, `tests/`: installation and decision-rule checks.

The model is MobileNetV3Small → global average pooling → Dense(128, ReLU) → Dropout(0.3) → Dense(25, softmax), with 1,016,201 parameters. It predicts 24 static letters and `nothing`. J/Z need motion; Space/Delete are outside the current label set.

Camera frames and uploaded photos travel to the app server. The application does not write them to disk or record video. Photo results and optional test-log entries stay in the current Streamlit session memory until cleared or the session ends. Downloaded crops are saved only when the user clicks Download. The hand model may be downloaded once from Google's official MediaPipe model bucket.

## Documentation

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- https://github.com/whitphx/streamlit-webrtc
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

See `VALIDATION.md` for checks actually performed and remaining deployment tests.
# ASL_Deep_Learning
