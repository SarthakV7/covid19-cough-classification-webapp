import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sound import sound
import SessionState
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components  # Import Streamlit

sample_rate = 48000
def load_wav(x, sample_rate=48000):
    '''This return the array values of audio with sampling rate of 48000 and Duration'''
    samples_, sample_rate = librosa.load(x, sr=sample_rate)
    non_silent = librosa.effects.split(samples_, frame_length=1024, hop_length=50)
    samples = np.concatenate([samples_[i:j] for i,j in non_silent])
    return samples

def pad_sample(x, max_length=220000):
  if len(x)<max_length:
    return np.hstack((x, np.zeros(max_length-len(x))))
  else:
    return x[:max_length]

def convert_to_mfccs(x, sr=48000):
  mfccs = librosa.feature.mfcc(x, n_mfcc=39, sr=sr)
  # mfccs = librosa.power_to_db(S=mfccs, ref=np.max)
  return mfccs

def convert_to_melspectrogram(x, sr=48000):
  mfccs = librosa.feature.melspectrogram(x, n_mels=64, sr=sr)
  mfccs = librosa.power_to_db(S=mfccs, ref=np.max)
  return mfccs

def convert_to_delta(mfcc):
    '''converting to velocity'''
    delta = librosa.feature.delta(mfcc)
    # delta = librosa.power_to_db(S=delta, ref=np.max)
    return delta

def convert_to_delta_2(mfcc):
    '''converting to velocity'''
    delta_2 = librosa.feature.delta(mfcc, order=2)
    # delta_2 = librosa.power_to_db(S=delta_2, ref=np.max)
    return delta_2

# Zero Crossing Rate
def ZCR(data):
  zcr = librosa.feature.zero_crossing_rate(data)
  return zcr[0]

# RMS energy
def RMSE(data):
  rmse = librosa.feature.rms(data)
  return rmse[0]

def standardize(data, axis=0):
  data -= np.mean(data, axis=axis)
  data /= np.std(data, axis=axis)
  return data

def expand_dims(X_train_mfcc):
  a,b,c = X_train_mfcc.shape
  res = np.zeros((a,b,c,3))
  for i in range(3):
    res[:,:,:,i] = X_train_mfcc
  return res

def process_data(file, individual=False):
  file_path = file
  x_aug = load_wav(file_path)
  x_aug_pad = pad_sample(x_aug)

  x_zcr = np.array([ZCR(x_aug_pad)[np.newaxis, ...]])
  x_rmse = np.array([RMSE(x_aug_pad)[np.newaxis, ...]])
  x_mfcc = np.array([convert_to_mfccs(x_aug_pad)])
  x_delta = convert_to_delta(x_mfcc)
  x_delta_2 = convert_to_delta_2(x_mfcc)
  x_melspectrogram = np.array([convert_to_melspectrogram(x_aug_pad)])

  x_img = np.concatenate((x_melspectrogram, x_mfcc, x_delta, x_delta_2, x_zcr, x_rmse), axis=1)
  x_img = expand_dims(x_img)
  return (x_aug, x_melspectrogram[0], x_mfcc[0], x_delta[0], x_delta_2[0], x_zcr[0][0], x_rmse[0][0]), x_img

# @st.cache
def load_model():
  model = tf.keras.models.load_model('./resnet50_covid.h5')
  return model

# st.title('COVID-19 classification through cough audio.')

# ***************************************************************************************************

def display_results(uploaded_file, flag='uploaded'):

    if flag=='uploaded':
        audio_bytes = uploaded_file.getvalue()
    elif flag=='recorded':
        audio_bytes = open(uploaded_file, 'rb').read()

    st.subheader('Sample of the submitted audio')
    st.audio(audio_bytes, format='audio/ogg')
    features, input_features = process_data(uploaded_file)
    output = model.predict(input_features)
    y_prob = np.round(output[0][0], 4)
    st.subheader('Status:')
    if y_prob>0.05:
        st.markdown('''<p style="font-size: 72px;
                        background: -webkit-linear-gradient(#eb3349, #f45c43);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-family: sans-serif;
                        font-weight: bold;
                        font-size:20px">
                        COVID-19 Positive.
                        </p>''', unsafe_allow_html=True)

    else:
        st.markdown('''<p style="font-size: 72px;
                        background: -webkit-linear-gradient(#56ab2f, #a8e063);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-family: sans-serif;
                        font-weight: bold;
                        font-size:20px">
                        COVID-19 Negative.
                        </p>''', unsafe_allow_html=True)
        st.balloons()

    # result = f'Probability of covid-19: {y_prob}'
    # st.subheader(result)
    st.subheader('Features used:')
    # option = st.selectbox('See the feaures extracetd from audio for testing:',
    #                      ("All","Audio wave" ,"Melspectrogram", "Mel-frequency cepstral coefficients (MFCCs)",
    #                       "∆ MFCCs", "∆-∆ MFCCs", "Zero Crossing Rate (ZCR)", "Root Mean Squared Energy (RMSE)"))

    (x_sample, x_melspectrogram, x_mfcc, x_delta, x_delta_2, x_zcr, x_rmse) = features

    sns.set(rc={'axes.facecolor':'#0E1117', 'figure.facecolor':'#0E1117', 'axes.grid':False})
    fig = plt.figure(figsize=(8,2))
    librosa.display.waveplot(x_sample, color='r', sr=48000)
    st.pyplot(fig)

    gap = int(np.ceil(len(x_sample)/512))
    time = int(np.ceil(len(x_sample)/48000))
    x_axis = np.linspace(0, time, gap)

    colorbar = dict(tickfont={'size':8})
    fig = make_subplots(rows=6, cols=1, subplot_titles=("Melspectrogram", "Mel-frequency cepstral coefficients (MFCCs)",
                                        "∆ MFCCs", "∆-∆ MFCCs", "Zero Crossing Rate (ZCR)", "Root Mean Squared Energy (RMSE)"))

    fig.add_trace(go.Heatmap(z=x_melspectrogram, x=x_axis, colorbar=colorbar, name='Melspectrogram'),
                  row=1, col=1)

    fig.add_trace(go.Heatmap(z=x_mfcc, x=x_axis, colorbar=colorbar, name='MFCCs'),
                  row=2, col=1)

    fig.add_trace(go.Heatmap(z=x_delta, x=x_axis, colorbar=colorbar, name='∆ MFCCs'),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=x_delta_2, x=x_axis, colorbar=colorbar, name='∆-∆ MFCCs'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(y=x_zcr, x=x_axis, name='ZCR',
                  mode='markers', marker_color=x_zcr),
                  row=5, col=1)

    fig.add_trace(go.Scatter(y=x_rmse, x=x_axis, name='RMSE',
                  mode='markers', marker_color=x_rmse),
                  row=6, col=1)

    fig.update_layout(height=1400, width=800)

    y_axes_labels = ['no. of mels']*4 + ['ZCR', 'RMSE']
    for idx, y in enumerate(y_axes_labels):
        fig['layout'][f'xaxis{idx+1}']['title'] = 'time'
        fig['layout'][f'yaxis{idx+1}']['title'] = y

    st.plotly_chart(fig)

# **************************************************************************************************

st.markdown('''<p style="font-size: 72px;
                background: -webkit-linear-gradient(#dd2476, #ff512f);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: sans-serif;
                font-weight: bold;
                font-size:40px">
                COVID-19 classification through cough audio.
                </p>''', unsafe_allow_html=True)

def small_title(x):
    text = f'''<p style="background: -webkit-linear-gradient(#dd2476, #ff512f);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-family: sans-serif;
                        font-weight: bold;
                        font-size:18px">
                        {x}
                        </p>'''
    return text

def html_links(text, link):
    return f'''<a href="{link}" target="_blank">{text}</a>'''

style = '''font_size: 14px;
           color: #aaa;'''

st.sidebar.title("About")
img_width = '60px'

text = f'''{small_title('The webapp')}
<p style="{style}">This webapp uses AI and cough audio to predict COVID-19.
Users can submit their cough audio either as a .wav file or directly record it on the dashboard.</p>
{small_title('The deep-learning model')}
<p style="{style}">The model was trained on crowd-sourced audio data made available by project {html_links('COSWARA', 'https://coswara.iisc.ac.in/about')} by IISC Bangalore and project {html_links('COUGHVID', 'https://coughvid.epfl.ch/about/')} by EPFL.
Although the model performs well and has Accuracy, Sensitivity, Specificity and, AUC score above 0.9, it still needs rigorous testing to reduce the False Negatives and Fasle Positives.</p>
{small_title('Cautionary advice')}
<p style="{style}">The model is still in beta-stage, and is purposely fine-tuned to be biased towards reducing the False Negatives which means, it is highly likely to classify a user as COVID-19 positive.
If there are any COVID-19 symptoms, please get tested irrespective of the classification result.</p>
{small_title('The developer')}
<p style="{style}">I am a data lover who loves to create impactful tools that could help people make this world a better place.</p>
<div>
<a href="https://github.com/SarthakV7/covid19-cough-classification-webapp" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/github.svg" width={img_width}"></a>
<a href="https://www.kaggle.com/sarthakvajpayee" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/kaggle.svg" width={img_width}"></a>
<a href="https://www.linkedin.com/in/sarthak-vajpayee/" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/linkedin.svg" width={img_width}"></a>
<a href="https://medium.com/@itssarthakvajpayee/" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/medium.png" width={img_width}"></a>
</div>
'''
st.sidebar.markdown(text, unsafe_allow_html=True)

data_load_state = st.text('Loading data...')
model = load_model()
data_load_state.text("Done! (using st.cache)")


st.subheader('Please submit the cough audio')
session_state = SessionState.get(name='', path=None)

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Choose a file... (Try to keep the audio short 5-6 seconds and upload as a .wav file)")
    submit_button_upl = st.form_submit_button(label='Submit the uploaded audio')

if st.button('Record'):
    with st.spinner(f'Recording for 5 seconds ....'):
        try:
            session_state.path = sound.record()
            st.write(session_state.path)
        except:
            pass
    st.success("Recording completed")

if st.button('Submit the recorded audio'):
    # filename = 'audio.wav'
    filename = session_state.path
    display_results(filename, flag='recorded')
    os.remove(filename)

if (uploaded_file is None and submit_button_upl):
    st.subheader('Something\'s not right, please refresh the page and retry!')

elif uploaded_file and submit_button_upl:
    display_results(uploaded_file, flag='uploaded')
