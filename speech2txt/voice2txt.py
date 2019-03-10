# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import io
import os

import keyboard

import speech2txt.record

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def run_quickstart():
    client = speech.SpeechClient()

    speech2txt.record.run(7)

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        'file.wav')

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    for result in response.results:
        keyboard.write(result.alternatives[0].transcript)
        print('Transcript: {}'.format(result.alternatives[0].transcript))


if __name__ == '__main__':
    run_quickstart()
