"""ASL classroom demo: browser camera, existing CNN, and clear feedback."""
from pathlib import Path
from io import BytesIO
from time import monotonic
import csv
import io
import hashlib
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
st.set_page_config(page_title='ASL | Alphabet recognition', page_icon='🤟', layout='wide')
st.title('ASL alphabet recognition')
st.caption('A deep learning mini-project · Static hand signs')


@st.cache_resource
def load_resources():
    model = SignModel(ROOT / 'model_results')
    path = get_hand_model(ROOT / 'hand_landmarker.task')
    model.predict(np.zeros((160, 160, 3), dtype=np.uint8))
    return model, path


with st.sidebar:
    st.header('Camera settings')
    mode = st.radio('Input', ['Live camera', 'Photo test'])
    crop_mode = st.selectbox('Hand framing', ['Automatic hand crop', 'Fixed guide box'])
    scale = st.slider('Wrist and background coverage', 1.2, 2.6, 1.8, 0.1,
                      disabled=crop_mode == 'Fixed guide box')
    flip = st.checkbox('Flip image before recognition', value=False,
                       help='Use only to compare camera/hand orientation. This changes model input.')
    with st.expander('Recognition settings'):
        threshold = st.slider('Minimum model score', 0.50, 0.95, 0.75, 0.05)
        gap = st.slider('Minimum gap between best two signs', 0.00, 0.40, 0.15, 0.05)
        st.caption('These are starting settings, not calibrated accuracy guarantees.')
    st.info('Show one hand with the wrist visible. Use even light and a plain background.')
    st.caption('A, K and V can need careful finger positioning. Compare with the reference images below.')

settings = Settings(crop_mode, scale, flip, threshold, gap)
st.caption('Camera frames are processed on the server. This app does not record video or save photos automatically.')
try:
    with st.spinner('Loading the classifier and hand detector…'):
        model, detector_path = load_resources()
except Exception as exc:
    st.error('The model could not start. Check the deployment logs or Internet connection and reload.')
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
        st.warning('Waiting for fresh video. Check the connection or restart the camera.')
        return
    st.metric('Recognized letter', result['label'] or '—')
    st.caption(result['status'])
    if result['crop'] is not None:
        st.image(result['crop'], caption='Exact 160 × 160 image seen by the CNN', width=200)
    if result['top']:
        st.dataframe([{'Candidate': name, 'Model score': f'{score:.1%}'}
                      for name, score in result['top']], hide_index=True, use_container_width=True)
        st.caption('Model scores are not measured real-world accuracy.')


left, right = st.columns([1.7, 1], gap='large')
if mode == 'Live camera':
    with left:
        st.subheader('Live camera')
        st.caption('Select your camera in the video controls, then press START. Microphone access is disabled.')
        ice_servers = [{'urls': ['stun:stun.l.google.com:19302']}]
        # Optional TURN credentials go into Streamlit Secrets, never GitHub.
        try:
            turn_url = st.secrets.get('TURN_URL', '')
        except FileNotFoundError:
            turn_url = ''
        if turn_url:
            ice_servers.append({'urls': [st.secrets['TURN_URL']],
                                'username': st.secrets['TURN_USERNAME'],
                                'credential': st.secrets['TURN_PASSWORD']})
        ctx = webrtc_streamer(
            key='asl-camera', mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(model, detector_path),
            rtc_configuration={'iceServers': ice_servers},
            media_stream_constraints={'video': {'width': {'ideal': 640},
                'height': {'ideal': 480}, 'frameRate': {'ideal': 10, 'max': 15}}, 'audio': False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.configure(settings)
        st.caption('If the camera stays on “connecting”, use Photo test. Some networks need a TURN relay; see README.')
    with right:
        st.subheader('Prediction')
        @st.fragment(run_every=0.3)
        def live_result():
            if ctx.state.playing and ctx.video_processor:
                show_result(ctx.video_processor.snapshot(), live=True)
            else:
                st.info('Press START to begin recognition.')
        live_result()
else:
    with left:
        st.subheader('Test a hand sign')
        source = st.radio('Photo source', ['Upload image', 'Take photo'], horizontal=True)
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
            st.image(st.session_state.photo_display, caption='Input and selected crop', use_container_width=True)
    with right:
        st.subheader('Prediction')
        result = st.session_state.get('photo_result') if photo is not None else None
        show_result(result)
        if result:
            actual = st.selectbox('Actual sign (for your test log)', model.names)
            if st.button('Add to test log'):
                rows = st.session_state.setdefault('test_log', [])
                rows.append({'actual': actual, 'predicted': result['top'][0][0] if result['top'] else '',
                    'accepted': result['label'] or '',
                    'score': result['top'][0][1] if result['top'] else 0,
                    'status': result['status'],
                    'crop_mode': crop_mode, 'crop_scale': scale, 'flip_input': flip,
                    'threshold': threshold, 'minimum_gap': gap})
                st.success('Added. Repeated entries count as separate trials.')
            if result['crop'] is not None:
                data = BytesIO()
                Image.fromarray(result['crop']).save(data, format='PNG')
                st.download_button('Download this test crop', data.getvalue(),
                                   file_name=f'{actual}_webcam_crop.png', mime='image/png')

with st.expander('Test log and real-world checks'):
    rows = st.session_state.get('test_log', [])
    if rows:
        correct = sum(r['actual'] == r['predicted'] for r in rows)
        accepted = [r for r in rows if r['accepted']]
        st.caption(f'Photos without a classifier prediction: {sum(not r["predicted"] for r in rows)}')
        st.write(f'{len(rows)} trials · Top-1 accuracy (missing detections count as failures): {correct/len(rows):.1%} · '
                 f'Accepted predictions: {len(accepted)}/{len(rows)}')
        if accepted:
            st.write(f'Accuracy among accepted predictions: '
                     f'{sum(r["actual"] == r["accepted"] for r in accepted)/len(accepted):.1%}')
        st.caption('Small, repeated or selected trials are not a formal accuracy evaluation.')
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button('Download test log', output.getvalue(), 'asl_test_log.csv', 'text/csv')
        if st.button('Clear test log'):
            st.session_state.test_log = []
            st.rerun()
    else:
        st.write('Use Photo test to record actual and predicted labels. Try different people, rooms and cameras.')
    st.write('Tune framing on one set of photos, then evaluate on separate, unseen photos. '
             'Include A, K, V, other similar signs, and no-sign examples. '
             'A higher rejection rate alone is not a better classifier.')

with st.expander('Reference signs and project details'):
    st.image(str(ROOT / 'model_results' / 'class_examples.png'),
             caption='Examples from the project dataset. Match the hand shape and orientation.')
    st.write('The saved MobileNetV3Small classifier recognizes 24 static letters plus “nothing”. '
             'J and Z need motion. Space and Delete are not output classes in this project.')
    st.write('Live recognition uses five agreeing predictions in a seven-frame window. '
             'Photo testing uses one prediction. Automatic framing includes wrist and background; '
             'the fixed guide box is a fallback if the hand detector misses a closed fist.')
    st.caption('The original dataset test accuracy was 99.02%. Webcam accuracy has not yet been established. '
               'The trained weights have not been changed for this web app.')
