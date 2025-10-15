# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
from .configuration_auto import AutoConfig
from .feature_extraction_auto import AutoFeatureExtractor
from .image_processing_auto import AutoImageProcessor
from .modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING,
    MODEL_FOR_BACKBONE_MAPPING,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_IMAGE_MAPPING,
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
    MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
    MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
    MODEL_FOR_MASK_GENERATION_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_RETRIEVAL_MAPPING,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TEXT_ENCODING_MAPPING,
    MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
    MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
    MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoBackbone,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDepthEstimation,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForImageTextToText,
    AutoModelForImageToImage,
    AutoModelForInstanceSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskedLM,
    AutoModelForMaskGeneration,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTextEncoding,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTokenClassification,
    AutoModelForUniversalSegmentation,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
    AutoModelWithLMHead,
)
from .processing_auto import AutoProcessor
