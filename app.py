"""ASL alphabet recognition: browser camera, CNN classifier and photo testing."""
from pathlib import Path
from io import BytesIO
from time import monotonic
import csv
import io
import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
# Load WebRTC native libraries before the ML stack to avoid symbol conflicts.
# model.py then selects the Keras backend before MediaPipe is imported.
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from model import SignModel
from hand_detector import get_hand_model
from recognition import Recognizer, Settings
from video import VideoProcessor

ROOT = Path(__file__).parent
STUN = {'urls': ['stun:stun.l.google.com:19302']}
st.set_page_config(page_title='ASL Alphabet Recognition', page_icon='🤟', layout='wide')
st.title('ASL Alphabet Recognition')
st.caption('Real-time recognition of static American Sign Language letters using a MobileNetV3 classifier.')


@st.cache_resource
def load_resources():
    model = SignModel(ROOT / 'model_results')
    path = get_hand_model(ROOT / 'hand_landmarker.task')
    model.predict(np.zeros((160, 160, 3), dtype=np.uint8))
    return model, path


def secret(name):
    try:
        return str(st.secrets.get(name, '')).strip()
    except FileNotFoundError:
        return ''


@st.cache_data(ttl=3600, show_spinner=False)
def cloudflare_ice_servers(key_id, api_token):
    request = urllib.request.Request(
        f'https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers',
        data=json.dumps({'ttl': 86400}).encode(), method='POST',
        # Cloudflare rejects Python's default User-Agent (error 1010), so set our own.
        headers={'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json',
                 'User-Agent': 'asl-recognition-app/1.0'})
    with urllib.request.urlopen(request, timeout=10) as response:
        servers = json.load(response)['iceServers']
    for server in servers:
        urls = server['urls'] if isinstance(server['urls'], list) else [server['urls']]
        # Port 53 entries time out in browsers; Cloudflare recommends dropping them.
        server['urls'] = [url for url in urls if not re.search(r':53(\?|$)', url)]
    return [server for server in servers if server['urls']]


def ice_servers():
    """Return (servers, status) where status is 'ok', 'missing' or 'failed'.

    Streamlit Cloud cannot accept direct WebRTC traffic, so a TURN relay is
    required there. STUN alone leaves the camera stuck after START.
    """
    if secret('CLOUDFLARE_TURN_KEY_ID') and secret('CLOUDFLARE_TURN_API_TOKEN'):
        try:
            return cloudflare_ice_servers(secret('CLOUDFLARE_TURN_KEY_ID'),
                                          secret('CLOUDFLARE_TURN_API_TOKEN')), 'ok'
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')[:200]
            logging.error('Cloudflare TURN request failed: HTTP %s %s', exc.code, detail)
            st.session_state.relay_error = f'Cloudflare HTTP {exc.code}: {detail}'
            return [STUN], 'failed'
        except Exception as exc:
            logging.exception('Could not get TURN credentials from Cloudflare')
            st.session_state.relay_error = f'{type(exc).__name__}: {exc}'
            return [STUN], 'failed'
    if secret('TURN_URL'):
        urls = [url.strip() for url in secret('TURN_URL').split(',') if url.strip()]
        return [STUN, {'urls': urls, 'username': secret('TURN_USERNAME'),
                       'credential': secret('TURN_PASSWORD')}], 'ok'
    return [STUN], 'missing'


with st.sidebar:
    st.header('Settings')
    mode = st.radio('Input', ['Live camera', 'Photo test'])
    crop_mode = st.selectbox('Hand framing', ['Automatic hand crop', 'Fixed guide box'])
    scale = st.slider('Crop padding', 1.2, 2.6, 1.8, 0.1,
                      disabled=crop_mode == 'Fixed guide box',
                      help='How much wrist and background to include around the detected hand.')
    flip = st.checkbox('Mirror input', value=False,
                       help='Flips the image before it reaches the model.')
    with st.expander('Advanced'):
        threshold = st.slider('Confidence threshold', 0.50, 0.95, 0.75, 0.05)
        gap = st.slider('Minimum margin over runner-up', 0.00, 0.40, 0.15, 0.05)
    st.divider()
    st.markdown('**Tips for best results**\n'
                '- One hand in frame, wrist visible\n'
                '- Even lighting\n'
                '- Plain background')

settings = Settings(crop_mode, scale, flip, threshold, gap)
try:
    with st.spinner('Loading model…'):
        model, detector_path = load_resources()
except Exception as exc:
    st.error('The model could not be loaded. Please reload the page.')
    with st.expander('Technical details'):
        st.code(str(exc))
    st.stop()


def show_result(result, live=False):
    if not result:
        st.info('Start the camera or submit a photo to see a prediction.')
        return
    if 'error' in result:
        st.error(result['error'])
        return
    if live and monotonic() - result['time'] > 2:
        st.warning('Waiting for video…')
        return
    st.metric('Recognized letter', result['label'] or '—')
    st.caption(result['status'])
    if result['crop'] is not None:
        st.image(result['crop'], caption='Model input (160 × 160)', width=200)
    if result['top']:
        st.dataframe([{'Letter': name, 'Confidence': f'{score:.1%}'}
                      for name, score in result['top']], hide_index=True, use_container_width=True)


left, right = st.columns([1.7, 1], gap='large')
if mode == 'Live camera':
    servers, relay = ice_servers()
    with left:
        st.subheader('Live camera')
        ctx = webrtc_streamer(
            key='asl-camera', mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(model, detector_path),
            rtc_configuration={'iceServers': servers},
            media_stream_constraints={'video': {'width': {'ideal': 640}, 'height': {'ideal': 480},
                                                'frameRate': {'ideal': 10}}, 'audio': False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.configure(settings)
        if relay == 'missing':
            st.warning('Live video relay is not configured, so the camera may not connect. '
                       'Use **Photo test** instead, or see the README to enable it.')
        elif relay == 'failed':
            st.warning('The live video relay could not be reached. Use **Photo test** '
                       'or try again shortly.')
            with st.expander('Technical details'):
                st.code(st.session_state.get('relay_error', 'Unknown error'))
    with right:
        st.subheader('Prediction')

        @st.fragment(run_every=0.3)
        def live_result():
            if ctx.state.playing and ctx.video_processor:
                show_result(ctx.video_processor.snapshot(), live=True)
            elif ctx.state.signalling:
                st.info('Connecting to camera…')
            else:
                st.info('Press **START** to begin.')
        live_result()
else:
    with left:
        st.subheader('Photo test')
        source = st.radio('Source', ['Upload image', 'Take photo'], horizontal=True)
        photo = (st.file_uploader('Choose a JPG or PNG', type=['jpg', 'jpeg', 'png'])
                 if source == 'Upload image' else st.camera_input('Take a photo'))
        if photo is not None:
            signature = hashlib.sha256(photo.getvalue() + repr(settings).encode()).hexdigest()
            if st.session_state.get('photo_signature') != signature:
                try:
                    with Image.open(photo) as original:
                        # Respect phone photo orientation before detecting the hand.
                        image = ImageOps.exif_transpose(original).convert('RGB')
                        image.thumbnail((1280, 1280))
                        rgb = np.asarray(image).copy()
                    recognizer = Recognizer(model, detector_path)
                    try:
                        display, result = recognizer.process(rgb, settings, use_voting=False)
                    finally:
                        recognizer.close()
                    st.session_state.update(photo_signature=signature, photo_result=result,
                                            photo_display=display)
                except Exception as exc:
                    st.error(f'Unable to process this photo: {exc}')
                    st.stop()
            st.image(st.session_state.photo_display, caption='Detected hand', use_container_width=True)
    with right:
        st.subheader('Prediction')
        result = st.session_state.get('photo_result') if photo is not None else None
        show_result(result)
        if result:
            actual = st.selectbox('Actual letter', model.names)
            if st.button('Add to test log'):
                rows = st.session_state.setdefault('test_log', [])
                rows.append({'actual': actual, 'predicted': result['top'][0][0] if result['top'] else '',
                    'accepted': result['label'] or '',
                    'score': result['top'][0][1] if result['top'] else 0,
                    'status': result['status'],
                    'crop_mode': crop_mode, 'crop_scale': scale, 'flip_input': flip,
                    'threshold': threshold, 'minimum_gap': gap})
                st.success('Added to test log.')
            if result['crop'] is not None:
                data = BytesIO()
                Image.fromarray(result['crop']).save(data, format='PNG')
                st.download_button('Download crop', data.getvalue(),
                                   file_name=f'{actual}_webcam_crop.png', mime='image/png')

with st.expander('Test log'):
    rows = st.session_state.get('test_log', [])
    if rows:
        correct = sum(r['actual'] == r['predicted'] for r in rows)
        accepted = [r for r in rows if r['accepted']]
        c1, c2, c3 = st.columns(3)
        c1.metric('Trials', len(rows))
        c2.metric('Top-1 accuracy', f'{correct/len(rows):.1%}')
        c3.metric('Accepted', f'{len(accepted)}/{len(rows)}')
        if accepted:
            st.caption(f'Accuracy among accepted predictions: '
                       f'{sum(r["actual"] == r["accepted"] for r in accepted)/len(accepted):.1%} · '
                       f'No hand detected: {sum(not r["predicted"] for r in rows)}')
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        b1, b2 = st.columns([1, 5])
        b1.download_button('Download CSV', output.getvalue(), 'asl_test_log.csv', 'text/csv')
        if b2.button('Clear log'):
            st.session_state.test_log = []
            st.rerun()
    else:
        st.caption('Use Photo test and choose the actual letter to record results here.')

with st.expander('Reference signs'):
    st.image(str(ROOT / 'model_results' / 'class_examples.png'),
             caption='Example of each supported letter from the training dataset.')

with st.expander('About the model'):
    st.markdown(
        '- **Architecture:** MobileNetV3Small → Dense(128) → Dense(25, softmax)\n'
        '- **Classes:** 24 static letters plus "nothing" (J and Z require motion and are not supported)\n'
        '- **Dataset test accuracy:** 99.02%\n'
        '- **Live mode:** a letter is shown once 5 of the last 7 frames agree\n'
        '- **Privacy:** frames are processed in memory and never stored')
