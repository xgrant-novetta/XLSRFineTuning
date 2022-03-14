# XLSRFineTuning

Based on the [Hugging Face competition](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) for transfer-learning with Facebook's Cross-Lingual Speech Recognition Wav2vec2 model. This is a script for training language models using transfer learning for languages with at least 500MB of data in the [Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset.


## How to Use

**Requirements**

Update and install dependencies with conda

'conda install -c conda-forge libsndfile librosa pydub'

Install pytorch versions if you need to from [here](https://pytorch.org/get-started/previous-versions/)

Install requirements

'pip install -r requirements.txt'

Before beginning, download the dataset from your language and make note of the folder location. The aduio will need to be converted to wav format and resampled to 16kHz. convert.py and resample.py are included

Edit the setup section in config.yml. The language code is from the CommonVoice dataset version. For example, English (version en_2181h_2020-12-11) has language code 'en'. Be sure to add the paths to your model and vocab.json file, and then any extra characters you want the trainer to ignore. Edit other configurations as needed.

Run 'python script.py' to begin.




