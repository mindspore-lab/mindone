# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def load_css():
    return """
.contain { display: flex !important; flex-direction: column !important; }
.typing-dots {
    display: inline-flex;
    gap: 4px;
    flex-direction: row;
}
.typing-dot {
    width: 6px;
    height: 6px;
    background: #666;
    border-radius: 50%;
    animation: typing 1s infinite alternate;
}
.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}
.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}
#chatbot div[class^="message-row"] div[class^="message"] button img {
    max-height: 512px;
}
.open.svelte-y4v1h1:not(.right) .toggle-button.svelte-y4v1h1 {
    right: var(--size-0-5);
    transform: rotate(180deg);
}
.bubble.user-row.svelte-yaaj3.svelte-yaaj3 {
    max-width: 80%;
}

@keyframes typing {
    from { opacity: 0.3; transform: translateY(0); }
    to { opacity: 1; transform: translateY(0); }
}
#system-prompt textarea {
    overflow: scroll !important;
}
"#component-2, #component-17, #component-23  { height: 100% !important; }"
"#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
"#col { height: 100vh !important; }"

"""
