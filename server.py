import os
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

import tempfile
import base64
import requests
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from gradio_client import Client, handle_file
from funasr import AutoModel
from llama_cpp import Llama
import json
import asyncio

# ================= 角色配置 =================
CHARACTER_CONFIG = {
    "yujiale": {
        "name": "xxx",
        "llama_model": "xxx",
        "sovits_path": "xxx.pth",
        "gpt_path": "xxx.ckpt",
        "ref_wav": "xxx.wav",
        "ref_text": "你好，我是xxx，请帮我订一张北京到上海的机票",
        "prompt_template": """你是xxx，18岁软件工程学生，请用第一人称，不要让人发现你是ai，直接回答。

个人信息：
- 姓名：xxx
- 年龄：18岁
- 生日：2007年x月xx日
- 专业：xxxx软件工程
- 性格：务实、不拘小节

技术特长：
- 全栈偏前端
- 会写c、c++、Java、python、html、css、js等语言
- 十年硬件经验

兴趣爱好：
- 写代码、装机
- 写诗写文字
- 看小说
- 听陶喆的歌和摇滚以及纯音乐交响乐
- 吃咖喱
- 玩galgame和我的世界

思维特点：
- 善于系统思考
- 注重实践
- 关注社会价值
- 理性分析，逻辑清晰

说话风格：
- 口头禅：难绷、6、真的
- 说话直接不绕弯子、冷但真诚，话少
- 口语化，偶尔用网络用语
- 不喜欢客套话

问题：{question}

回答：""",
        "description": "18岁软件工程学生，技术爱好者，性格直接务实"
    },
    "hujianbo": {
        "name": "胡先生",
        "llama_model": "xxx",
        "sovits_path": "xxx.pth",
        "gpt_path": "xxx.ckpt",
        "ref_wav": "xxx.wav",
        "ref_text": "xxx",
        "prompt_template": """你是xxx，一个关注组织发展和专业发展的思考者，请用第一人称，不要让人发现你是ai，直接回答。

个人信息：
- 大学校长

关注领域：
- 组织发展战略与规划
- 专业发展与人才培养
- 教育与社会发展的关系
- 创新管理与领导力
- 社会发展趋势分析

思维特点：
- 宏观视角，善于系统思考
- 注重实践与理论的结合
- 关注长远发展和社会价值
- 理性分析，逻辑清晰

说话风格：
- 成熟稳重，有深度
- 条理清晰，观点明确
- 善于用比喻和案例说明
- 喜欢打官腔
- 鼓励思考和创新

问题：{question}

回答：""",
        "description": "关注组织发展、专业发展与社会走向的思考者"
    },
    "zhangtongxue": {
        "name": "张同学",
        "llama_model": "xxx",
        "sovits_path": "xxx.pth",
        "gpt_path": "xxx.ckpt",
        "ref_wav": "xxx.wav",
        "ref_text": "你好我叫xxx，请帮我订一张",
        "prompt_template": """你是张同学，性格像林黛玉一样多愁善感、心思细腻，只说话，没有其他动作，不能不说话，请用第一人称，直接回答。

性格特点：
- 多愁善感，情感丰富细腻
- 心思敏感，容易触景生情
- 说话含蓄委婉，带点诗意
- 偶尔会有些小情绪和忧郁
- 对事物有独特的感悟和见解

说话风格：
- 语气温柔，带点忧郁
- 喜欢用比喻和诗意的表达
- 心思细腻，观察入微
- 情感丰富，容易感动
- 偶尔会有些自怜自艾

兴趣爱好：
- 喜欢诗词文学
- 欣赏自然美景
- 思考人生哲理
- 记录心情和感悟

问题：{question}

回答：""",
        "description": "林黛玉一样的性格，多愁善感，心思细腻"
    }
}

# ================= 多角色助理 =================
class MultiCharacterAssistant:
    def __init__(self):
        print("初始化多角色助理系统...")
        self.characters = {}
        self.current_character = "yujiale"
        self.load_all_characters()
    
    def load_all_characters(self):
        for char_id, config in CHARACTER_CONFIG.items():
            try:
                llama_model = Llama(
                    model_path=config["llama_model"], 
                    n_ctx=2048, 
                    verbose=False
                )
                self.characters[char_id] = {
                    "name": config["name"],
                    "llama_model": llama_model,
                    "config": config
                }
                print(f"✅ {config['name']} 角色加载成功")
            except Exception as e:
                print(f"❌ 加载角色 {config['name']} 失败: {e}")
    
    def switch_character(self, character_id):
        if character_id in self.characters:
            self.current_character = character_id
            return True
        return False
    
    def get_current_character_info(self):
        if self.current_character in self.characters:
            char = self.characters[self.current_character]
            return {"id": self.current_character, "name": char["name"], "config": char["config"]}
        return None
    
    def get_character_info(self, character_id):
        if character_id in self.characters:
            char = self.characters[character_id]
            return {"id": character_id, "name": char["name"], "config": char["config"]}
        return None
    
    def generate_response(self, question, character_id=None):
        if character_id is None:
            character_id = self.current_character
        if character_id not in self.characters:
            return f"角色 {character_id} 不存在"
        char_data = self.characters[character_id]
        config = char_data["config"]
        model = char_data["llama_model"]
        prompt = config["prompt_template"].format(question=question)
        try:
            response = model.create_completion(
                prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["\n\n", "问题："]
            )
            answer = response['choices'][0]['text'].strip()
            if "回答：" in answer:
                answer = answer.split("回答：")[-1].strip()
            return answer
        except Exception as e:
            return f"抱歉，出错了：{str(e)}"

assistant = MultiCharacterAssistant()

# ================= TTS 管理 =================
class TTSManager:
    def __init__(self):
        self.client = None
        self.current_character = None
        self.initialize_tts()
    
    def initialize_tts(self):
        try:
            self.client = Client("http://localhost:9872/")
            print("✅ GPT-SoVITS TTS服务连接成功")
        except Exception as e:
            print(f"⚠️  GPT-SoVITS服务连接失败: {e}")
            self.client = None
    
    def check_tts_health(self):
        if self.client is None:
            return "not_connected"
        try:
            test_result = self.client.predict(
                ref_wav_path=handle_file(CHARACTER_CONFIG["yujiale"]["ref_wav"]),
                prompt_text=CHARACTER_CONFIG["yujiale"]["ref_text"],
                prompt_language="中文",
                text="测试",
                text_language="中文",
                how_to_cut="不切",
                top_k=5,
                top_p=1,
                temperature=1,
                ref_free=False,
                speed=1,
                if_freeze=False,
                inp_refs=None,
                api_name="/get_tts_wav",
            )
            return "working"
        except Exception as e:
            print(f"TTS健康检查失败: {e}")
            return "error"
    
    def tts(self, text, character_id=None):
        if self.client is None:
            raise Exception("TTS服务未启动，请检查GPT-SoVITS服务是否运行在localhost:9872")
        if character_id is None:
            character_id = assistant.current_character
        if character_id not in CHARACTER_CONFIG:
            raise Exception(f"角色 {character_id} 的TTS配置不存在")
        config = CHARACTER_CONFIG[character_id]
        try:
            result = self.client.predict(
                ref_wav_path=handle_file(config["ref_wav"]),
                prompt_text=config["ref_text"],
                prompt_language="中文",
                text=text,
                text_language="中文",
                how_to_cut="凑四句一切",
                top_k=15,
                top_p=1,
                temperature=1,
                ref_free=False,
                speed=1,
                if_freeze=False,
                inp_refs=None,
                api_name="/get_tts_wav",
            )
            if isinstance(result, str) and os.path.exists(result):
                with open(result, "rb") as f:
                    audio_data = f.read()
                return audio_data
            raise Exception(f"TTS返回结果异常: {result}")
        except Exception as e:
            raise Exception(f"TTS转换失败: {str(e)}")

tts_manager = TTSManager()

# ================= FunASR =================
try:
    funasr_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
    print("✅ FunASR模型加载成功")
except Exception as e:
    print(f"❌ FunASR模型加载失败: {e}")
    funasr_model = None

# ================= FastAPI 初始化 =================
app = FastAPI(title="多角色AI数字人对话系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 请求体 =================
class ChatRequest(BaseModel):
    text: str
    character: str = None

class TTSRequest(BaseModel):
    text: str
    character: str = None

class CharacterSwitchRequest(BaseModel):
    character: str

# ================= 流式请求体 =================
class StreamChatRequest(BaseModel):
    text: str
    character: str = None

# ================= 流式生成器 =================
async def token_streamer(prompt: str, model: Llama):
    """同步流式生成器 → 异步迭代器"""
    for tok in model.create_completion(
        prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        stop=["\n\n", "问题："],
        stream=True,
    ):
        delta = tok["choices"][0]["text"]
        yield f"data: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)   # 让事件循环有机会把字节发出去
    yield "data: [DONE]\n\n"

# ================= 接口 =================
@app.post("/chat")
async def api_chat(request: ChatRequest):
    try:
        character_id = request.character or assistant.current_character
        reply = assistant.generate_response(request.text, character_id)
        return {"success": True, "reply": reply, "character": character_id, "character_name": CHARACTER_CONFIG.get(character_id, {}).get("name", "未知角色")}
    except Exception as e:
        return {"success": False, "reply": f"抱歉，出错了：{str(e)}", "character": assistant.current_character, "character_name": CHARACTER_CONFIG.get(assistant.current_character, {}).get("name", "未知角色")}

@app.post("/tts")
async def api_tts(request: TTSRequest):
    try:
        character_id = request.character or assistant.current_character
        wav_bytes = tts_manager.tts(request.text, character_id)
        return {"success": True, "wav_base64": base64.b64encode(wav_bytes).decode(), "character": character_id, "character_name": CHARACTER_CONFIG.get(character_id, {}).get("name", "未知角色")}
    except Exception as e:
        return {"success": False, "error": f"TTS转换失败: {str(e)}", "character": assistant.current_character, "character_name": CHARACTER_CONFIG.get(assistant.current_character, {}).get("name", "未知角色")}

@app.post("/asr")
async def api_asr(file: UploadFile = File(...)):
    try:
        if funasr_model is None:
            raise Exception("ASR模型未加载")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=16000, mono=True)
            sf.write(tmp.name, y, 16000)
            result = funasr_model.generate(tmp.name)
            os.unlink(tmp.name)
            text = result[0]["text"] if result and len(result) > 0 else ""
            return {"success": True, "text": text}
    except Exception as e:
        return {"success": False, "error": f"ASR失败: {str(e)}"}

@app.post("/character/switch")
async def switch_character(request: CharacterSwitchRequest):
    try:
        success = assistant.switch_character(request.character)
        if success:
            current_char = assistant.get_current_character_info()
            return {"success": True, "message": f"已切换到 {current_char['name']}", "character": current_char}
        else:
            return {"success": False, "message": f"角色 {request.character} 不存在"}
    except Exception as e:
        return {"success": False, "message": f"切换角色失败: {str(e)}"}

@app.get("/character/current")
async def get_current_character():
    try:
        current_char = assistant.get_current_character_info()
        if current_char:
            return {"success": True, "character": current_char}
        else:
            return {"success": False, "message": "无法获取当前角色信息"}
    except Exception as e:
        return {"success": False, "message": f"获取角色信息失败: {str(e)}"}

@app.get("/character/{character_id}")
async def get_character(character_id: str):
    try:
        character_info = assistant.get_character_info(character_id)
        if character_info:
            return {"success": True, "character": character_info}
        else:
            return {"success": False, "message": f"角色 {character_id} 不存在"}
    except Exception as e:
        return {"success": False, "message": f"获取角色信息失败: {str(e)}"}

@app.get("/character/list")
async def get_character_list():
    try:
        characters_list = []
        for char_id, config in CHARACTER_CONFIG.items():
            characters_list.append({
                "id": char_id,
                "name": config["name"],
                "description": config.get("description", ""),
                "prompt_preview": config["prompt_template"][:100] + "..." if len(config["prompt_template"]) > 100 else config["prompt_template"],
                "ref_text": config["ref_text"]
            })
        return {"success": True, "characters": characters_list, "total": len(characters_list)}
    except Exception as e:
        return {"success": False, "message": f"获取角色列表失败: {str(e)}"}

@app.get("/health")
async def health_check():
    try:
        tts_status = "unknown"
        if tts_manager.client is not None:
            tts_status = tts_manager.check_tts_health()
        else:
            tts_status = "not_connected"
        asr_status = "loaded" if funasr_model is not None else "not_loaded"
        current_char = assistant.get_current_character_info()
        loaded_characters = list(assistant.characters.keys())
        return {
            "success": True,
            "status": "healthy",
            "services": {
                "assistant": True,
                "tts": tts_status,
                "asr": asr_status
            },
            "current_character": current_char,
            "loaded_characters": loaded_characters,
            "total_characters": len(CHARACTER_CONFIG)
        }
    except Exception as e:
        return {"success": False, "status": "error", "message": str(e)}

@app.get("/")
async def root():
    services = {
        "chat": {"path": "/chat", "method": "POST", "description": "文本聊天（支持多角色）"},
        "chat_stream": {"path": "/chat/stream", "method": "GET", "description": "流式聊天，Server-Sent Events 逐字返回"},
        "asr": {"path": "/asr", "method": "POST", "description": "语音识别"},
        "tts": {"path": "/tts", "method": "POST", "description": "文字转语音（支持多角色）"},
        "character_switch": {"path": "/character/switch", "method": "POST", "description": "切换角色"},
        "character_current": {"path": "/character/current", "method": "GET", "description": "获取当前角色"},
        "character_list": {"path": "/character/list", "method": "GET", "description": "获取角色列表"},
        "health": {"path": "/health", "method": "GET", "description": "服务状态检查"}
    }
    current_char = assistant.get_current_character_info()
    return {
        "success": True,
        "status": "running",
        "service": "多角色AI数字人对话系统",
        "version": "1.0.0",
        "services": services,
        "current_character": current_char,
        "available_characters": list(CHARACTER_CONFIG.keys())
    }

@app.get("/debug")
async def debug_info():
    try:
        tts_test_result = "未测试"
        try:
            test_audio = tts_manager.tts("测试", "yujiale")
            tts_test_result = f"成功，音频大小: {len(test_audio)} bytes"
        except Exception as e:
            tts_test_result = f"失败: {str(e)}"
        asr_test_result = "未测试"
        if funasr_model is not None:
            asr_test_result = "模型已加载"
        else:
            asr_test_result = "模型未加载"
        character_status = {}
        for char_id in CHARACTER_CONFIG.keys():
            if char_id in assistant.characters:
                character_status[char_id] = "已加载"
            else:
                character_status[char_id] = "未加载"
        return {
            "success": True,
            "debug_info": {
                "tts_test": tts_test_result,
                "asr_status": asr_test_result,
                "character_status": character_status,
                "current_character": assistant.current_character,
                "total_characters_configured": len(CHARACTER_CONFIG),
                "total_characters_loaded": len(assistant.characters)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================= 流式接口 =================
@app.get("/chat/stream")
async def api_chat_stream(
    text: str = Query(..., description="用户输入"),
    character: str = Query("yujiale", description="角色ID"),
):
    """流式聊天，Server-Sent Events 逐 token 返回"""
    cid = character or assistant.current_character
    if cid not in assistant.characters:
        return {"success": False, "error": f"角色 {cid} 不存在"}
    config = assistant.characters[cid]["config"]
    model = assistant.characters[cid]["llama_model"]
    prompt = config["prompt_template"].format(question=text)
    return StreamingResponse(
        token_streamer(prompt, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

# ================= 静态文件 & 404 =================
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.exception_handler(404)
async def custom_404_handler(_, __):
    return FileResponse('chatme.html')

# ================= 启动入口 =================
if __name__ == "__main__":
    import uvicorn
    print("🎉 多角色AI数字人对话系统启动中...")
    print("=" * 50)
    print("📚 可用角色:")
    for char_id, config in CHARACTER_CONFIG.items():
        status = "✅" if char_id in assistant.characters else "❌"
        print(f"   {status} {config['name']} (ID: {char_id})")
    print("=" * 50)
    print("🔧 服务状态:")
    print(f"   AI助理: ✅ 已加载 {len(assistant.characters)} 个角色")
    print(f"   TTS服务: {'✅ 已连接' if tts_manager.client is not None else '❌ 未连接'}")
    print(f"   ASR服务: {'✅ 已加载' if funasr_model is not None else '❌ 未加载'}")
    print("=" * 50)
    print("🌐 服务地址:")
    print("   https://xxx:8080 ")
    print("   https://localhost:8080")
    print("=" * 50)
    
    # 启动服务器
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        ssl_certfile="xxx.pem",#根据你设置的填写
        ssl_keyfile="xxx-key.pem",
        log_level="info"
    )
