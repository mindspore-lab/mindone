save_dir='example/results'
model_dir="pretrained_models/Spark-TTS-0.5B"
text="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
prompt_text="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
prompt_speech_path="example/prompt_audio.wav"

# Run inference
python -m cli.inference \
    --text "${text}" \
    --save_dir "${save_dir}" \
    --model_dir "${model_dir}" \
    --prompt_text "${prompt_text}" \
    --prompt_speech_path "${prompt_speech_path}" \

    ### control arguments
    # --gender female \
    # --pitch very_low \
    # --speed very_high \
