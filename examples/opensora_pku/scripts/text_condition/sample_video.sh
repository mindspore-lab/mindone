python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0/65x512x512 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --captions examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --guidance_scale 7.5 \
    --sampling_steps 250
