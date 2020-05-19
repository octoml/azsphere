## Keyword Spotting Model
### Before running the model

- Download [ARM repository] (https://github.com/ARM-software/ML-KWS-for-MCU) in home directory
- Export Env variable to this directory
```bash
cd $HOME
git clone https://github.com/ARM-software/ML-KWS-for-MCU
export ARM_KWS_PATH="${HOME}/ML-KWS-for-MCU"
echo $ARM_KWS_PATH
```

### Export model
```bash
python3 keyword_spotting.py --export --quantize --global-scale 4.0 --debug
```

### Accuracy evaluation
```bash
python3 keyword_spotting.py --test --quantize --module keyword_model/module_gs_4.0.pickle
```
