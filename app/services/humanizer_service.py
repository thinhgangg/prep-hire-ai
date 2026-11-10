from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm_provider_service import LLMProviderError, LLMProviderService

logger = logging.getLogger(__name__)

# Common Vietnamese speech fillers, hesitation markers, and conversational connectors
VIETNAMESE_FILLERS = (
    "ừm", "ờ", "à", "ạ", "dạ", "thì", "kiểu", "kiểu như", "kiểu là",
    "nói chung là", "thực ra thì", "thật ra thì", "thật sự là",
    "em nghĩ là", "theo em thì", "em thấy là",
    "với lại", "xong rồi", "xong thì", "rồi các thứ", "mấy cái",
    "cơ mà", "đấy", "nhỉ", "đúng không", "đại loại là", "nói thật là"
)

SELF_CORRECTION_PATTERNS = (
    "à không", "ý em là", "ý tôi là", "nói đúng hơn là",
    "nhầm một chút", "à nhầm", "hoặc là như", "nói chính xác thì"
)

PERSONA_PROMPTS = {
    "nervous_junior": {
        "name": "Ứng viên Junior / Thực tập sinh (Hơi rụt rè, ngập ngừng, thật thà)",
        "description": (
            "Ứng viên mới tốt nghiệp hoặc ít kinh nghiệm. Có tâm lý hồi hộp khi phỏng vấn, "
            "dùng nhiều từ đệm ('dạ...', 'ừm...', 'thực ra thì...'), có nhiều dấu ngắt quãng '...' khi suy nghĩ, "
            "thỉnh thoảng tự sửa lời ('à không, ý em là...'). Kể lại trải nghiệm từ đồ án, bài tập lớn hoặc "
            "dự án thực tập một cách mộc mạc, không dùng gạch đầu dòng, không hoàn hảo như sách giáo khoa."
        ),
        "guidelines": (
            "- Bắt đầu bằng từ đệm tự nhiên như: 'Dạ... ừm...', 'Thực ra thì...'\n"
            "- Thêm 2-4 khoảng lặng '...' thể hiện đang lục tìm trí nhớ.\n"
            "- Có ít nhất 1 lần tự đính chính: 'à không, ý em là...', 'nhầm một chút, lúc đó là...'\n"
            "- Thừa nhận một giới hạn thực tế: 'đợt đó em làm chưa tối ưu lắm nhưng...', 'lúc đầu em cũng hơi lúng túng...'\n"
            "- Kết thúc bằng câu dò hỏi nhẹ nhàng hoặc cảm ơn: '...dạ em nghĩ là vậy ạ', '...đại loại thế ạ'."
        )
    },
    "confident_mid": {
        "name": "Kỹ sư Middle 2-3 năm (Thực chiến, tự tin, văn nói công sở)",
        "description": (
            "Kỹ sư đã đi làm thực tế, phong thái tự tin, phản xạ nhanh nhưng vẫn là văn nói đời thường. "
            "Dùng từ ngữ thực chiến ('đợt đó bên em...', 'vấn đề lúc đấy là...', 'trade-off là...'), "
            "có nhịp ngắt khi cân nhắc các phương án, dùng tiếng bồi công nghệ tự nhiên nhưng giải thích mạch lạc."
        ),
        "guidelines": (
            "- Giọng điệu chủ động, thực tế: 'Đợt đó bên em triển khai...', 'Về câu này thì theo kinh nghiệm của em...'\n"
            "- Có ngắt nhịp suy nghĩ về kiến trúc/số liệu: '...lúc đó đo đạc thấy latency tầm...'\n"
            "- Đề cập trade-off thực tế thay vì lý thuyết suông: 'nếu dùng cách A thì tốn RAM nên bên em chốt cách B...'\n"
            "- Giữ văn nói tự nhiên, tuyệt đối không dùng gạch đầu dòng 1, 2, 3."
        )
    },
    "hesitant_struggling": {
        "name": "Ứng viên ấp úng, ngập ngừng cao (Gặp câu hỏi khó, suy nghĩ tại chỗ)",
        "description": (
            "Ứng viên gặp câu hỏi kỹ thuật hóc búa hoặc bất ngờ, phải vừa nói vừa suy nghĩ. "
            "Tần suất ngập ngừng rất cao ('ờ...', 'ừm...', '...'), lặp lại từ do chưa kịp nghĩ ra từ tiếp theo "
            "('thì lúc đó thì lúc đó...', 'em dùng em dùng...'), cố gắng diễn đạt ý niệm bằng lời mộc mạc."
        ),
        "guidelines": (
            "- Mật độ ngập ngừng cao với nhiều '...', 'ừm...', 'ờ...'\n"
            "- Có hiện tượng lặp từ khi bí từ: 'trong cái... trong cái phần xử lý...', 'bọn em phải... phải tìm cách...'\n"
            "- Thể hiện sự đắn đo: 'Để em nhớ lại xem...', 'Cái này... ừm... thật ra em nhớ mang máng là...'\n"
            "- Diễn đạt logic cốt lõi nhưng câu từ có phần lộn xộn đúng kiểu người đang chịu áp lực phòng phỏng vấn."
        )
    },
    "conversational_casual": {
        "name": "Giao tiếp tự nhiên, đời thường (Thoải mái, ví von gần gũi)",
        "description": (
            "Ứng viên trả lời theo phong cách trò chuyện cởi mở, dùng nhiều quán ngữ giao tiếp tiếng Việt "
            "('kiểu như là...', 'nói chung là...', 'xong rồi...', 'cơ bản thì...'). Giải thích kỹ thuật "
            "theo lối trực quan, dễ hiểu như đang giải thích cho đồng nghiệp cùng team."
        ),
        "guidelines": (
            "- Dùng từ nối khẩu ngữ: 'kiểu như là', 'nói chung là', 'xong rồi', 'mấy cái vụ này...'\n"
            "- Ví von hoặc so sánh đời thường: 'nó giống như là mình xếp hàng...', 'thay vì gọi liên tục thì mình gom lại...'\n"
            "- Nhịp nói trôi chảy, thân thiện, không quan cách."
        )
    },
    "stt_spoken_real": {
        "name": "Mô phỏng lời nói thu âm qua Micro STT (Chân thực 100%)",
        "description": (
            "Tái hiện chính xác những gì hệ thống Speech-to-Text (như Whisper, Google STT) ghi nhận từ micro "
            "của ứng viên thật: câu không có cấu trúc ngữ pháp hoàn chỉnh, có hiện tượng lặp từ khi nói nhanh, "
            "ngắt câu theo hơi thở, từ mượn tiếng Anh được nói tự nhiên."
        ),
        "guidelines": (
            "- Không chấm phẩy chuẩn mực theo ngữ pháp sách giáo khoa, ngắt nhịp thở bằng dấu phẩy hoặc '...'\n"
            "- Xuất hiện lặp từ ngẫu nhiên: 'trong dự án trong dự án...', 'em có làm em có làm qua...'\n"
            "- Có những đoạn chuyển ý đột ngột khi ứng viên vừa nghĩ ra ý mới.\n"
            "- Câu từ chân thực như trích từ file audio ghi âm thật."
        )
    }
}


@dataclass
class HumanizeMetrics:
    word_count: int
    filler_count: int
    pause_count: int
    self_correction_count: int
    hesitation_score: float  # Scale 0 - 100 representing conversational realism

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanizedAnswer:
    persona: str
    persona_name: str
    answer_text: str
    metrics: HumanizeMetrics
    provider: str
    model: str
    latency_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "persona_name": self.persona_name,
            "answer_text": self.answer_text,
            "metrics": self.metrics.as_dict(),
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


@dataclass
class MultiPersonaResult:
    question_text: str
    context_text: Optional[str]
    original_ai_answer: Optional[str]
    answers: Dict[str, HumanizedAnswer] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_text": self.question_text,
            "context_text": self.context_text,
            "original_ai_answer": self.original_ai_answer,
            "answers": {k: v.as_dict() for k, v in self.answers.items()},
        }


class HumanizedAnswerService:
    """
    Service generating human-like spoken interview answers with emotional tones,
    hesitations, pauses, filler words, and self-corrections in Vietnamese.
    """

    PROVIDERS_PRIORITY = ("groq", "cloudflare", "inception", "opentyphoon", "ollama_cloud")

    @classmethod
    def calculate_metrics(cls, text: str) -> HumanizeMetrics:
        """
        Calculates linguistic features reflecting spoken human conversational speech.
        """
        normalized = text.lower()
        words = re.findall(r"\b\w+\b", normalized, flags=re.UNICODE)
        word_count = len(words)

        if word_count == 0:
            return HumanizeMetrics(0, 0, 0, 0, 0.0)

        filler_count = 0
        for filler in VIETNAMESE_FILLERS:
            # Count whole phrase / word matches
            filler_count += len(re.findall(r"\b" + re.escape(filler) + r"\b", normalized))

        pause_count = len(re.findall(r"\.{2,}|…", text))

        self_correction_count = 0
        for pat in SELF_CORRECTION_PATTERNS:
            self_correction_count += len(re.findall(r"\b" + re.escape(pat) + r"\b", normalized))

        # Hesitation score formula based on density of speech traits
        filler_density = (filler_count / word_count) * 100
        pause_density = (pause_count / max(1, word_count // 20)) * 10
        correction_bonus = self_correction_count * 15

        hesitation_score = min(100.0, max(0.0, round(filler_density * 4 + pause_density * 5 + correction_bonus, 1)))

        return HumanizeMetrics(
            word_count=word_count,
            filler_count=filler_count,
            pause_count=pause_count,
            self_correction_count=self_correction_count,
            hesitation_score=hesitation_score,
        )

    @classmethod
    def _execute_llm_json(
        cls,
        prompt: str,
        preferred_provider: Optional[str] = None,
        operation: str = "generate_humanized_answer",
        temperature: float = 0.75,
    ) -> Tuple[dict[str, Any], str, str, int]:
        """
        Executes prompt across configured providers with automatic fallback.
        """
        candidates = list(cls.PROVIDERS_PRIORITY)
        if preferred_provider and preferred_provider in candidates:
            candidates.remove(preferred_provider)
            candidates.insert(0, preferred_provider)

        last_error = None
        for provider in candidates:
            try:
                res = LLMProviderService.generate_json(
                    provider,
                    prompt,
                    operation=operation,
                    temperature=temperature,
                    max_output_tokens=2500,
                )
                return res.data, res.provider, res.model, res.latency_ms
            except Exception as exc:
                last_error = exc
                logger.warning("Humanizer fallback: provider %s failed (%s), trying next...", provider, exc)
                continue

        raise RuntimeError(f"All LLM providers failed for humanizer: {last_error}")

    @classmethod
    def generate_single_persona(
        cls,
        question: str,
        context: Optional[str] = None,
        persona: str = "nervous_junior",
        preferred_provider: Optional[str] = None,
        temperature: float = 0.75,
    ) -> HumanizedAnswer:
        """
        Generates a humanized interview answer for a single persona.
        """
        persona_info = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["nervous_junior"])
        prompt = f"""Bạn là một chuyên gia ngôn ngữ học chuyên mô phỏng hội thoại phỏng vấn xin việc tiếng Việt thực tế.
Nhiệm vụ: Tạo một câu trả lời phỏng vấn BẰNG GIỌNG NÓI TỰ NHIÊN CỦA CON NGƯỜI (Spoken Natural Speech), mô phỏng đúng tính cách ứng viên.

CÂU HỎI PHỎNG VẤN:
"{question}"

{f'BỐI CẢNH ỨNG VIÊN / DỰ ÁN (NẾU CÓ): {context}' if context else ''}

PHONG CÁCH ỨNG VIÊN CẦN MÔ PHỎNG:
- Tên phong cách: {persona_info['name']}
- Mô tả: {persona_info['description']}
- Hướng dẫn cụ thể:
{persona_info['guidelines']}

QUY TẮC BẮT BUỘC (CRITICAL):
1. TUYỆT ĐỐI KHÔNG dùng gạch đầu dòng, danh sách đánh số 1. 2. 3. hay văn phong AI chuẩn mực.
2. Dùng 100% văn nói tiếng Việt (Spoken Vietnamese), có từ đệm ("ừm...", "dạ...", "thì..."), có ngập ngừng suy nghĩ ("..."), có cảm xúc thật.
3. Câu trả lời phải trả lời đúng trọng tâm kỹ thuật nhưng diễn đạt dưới dạng lời nói miệng trực tiếp khi ngồi trước nhà tuyển dụng.
4. Trả về định dạng JSON thuần túy:
{{
  "answer": "Nội dung câu trả lời nói miệng tự nhiên ở đây..."
}}
"""
        started = time.perf_counter()
        data, provider, model, _ = cls._execute_llm_json(
            prompt,
            preferred_provider=preferred_provider,
            operation=f"humanize_{persona}",
            temperature=temperature,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        answer_text = str(data.get("answer") or "").strip()
        metrics = cls.calculate_metrics(answer_text)

        return HumanizedAnswer(
            persona=persona,
            persona_name=persona_info["name"],
            answer_text=answer_text,
            metrics=metrics,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )

    @classmethod
    def humanize_existing_answer(
        cls,
        question: str,
        raw_ai_answer: str,
        persona: str = "nervous_junior",
        preferred_provider: Optional[str] = None,
        temperature: float = 0.75,
    ) -> HumanizedAnswer:
        """
        Transforms an existing robotic/academic AI answer into a natural spoken human response.
        """
        persona_info = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["nervous_junior"])
        prompt = f"""Bạn là một chuyên gia ngôn ngữ học tiếng Việt.
Dưới đây là một câu trả lời phỏng vấn được viết bởi AI (mang nặng tính văn viết, cấu trúc hoàn hảo, thiếu tự nhiên).
Nhiệm vụ của bạn là VIẾT LẠI câu trả lời này thành LỜI NÓI THẬT CỦA CON NGƯỜI (Spoken Vietnamese), mô phỏng một ứng viên thực tế đang trả lời bằng giọng nói qua micro.

CÂU HỎI PHỎNG VẤN:
"{question}"

CÂU TRẢ LỜI GỐC CỦA AI (QUÁ CHỈNH CHU):
"{raw_ai_answer}"

PHONG CÁCH CẦN CHUYỂN ĐỔI THÀNH:
- Tên phong cách: {persona_info['name']}
- Đặc điểm: {persona_info['description']}
- Yêu cầu:
{persona_info['guidelines']}

QUY TẮC CẦN TUÂN THỦ:
1. Giữ nguyên các ý kỹ thuật cốt lõi trong câu trả lời gốc, nhưng PHÁ VỠ cấu trúc bài luận và các gạch đầu dòng của AI.
2. Chuyển thành văn nói tự nhiên, có ngập ngừng ("ừm...", "dạ...", "..."), có chỗ giải thích bằng lời mộc mạc, có thể tự đính chính lời nói.
3. Không thêm các lời giải thích thừa ngoài câu trả lời.
4. Trả về đúng định dạng JSON:
{{
  "answer": "Nội dung câu trả lời bằng văn nói người thật..."
}}
"""
        started = time.perf_counter()
        data, provider, model, _ = cls._execute_llm_json(
            prompt,
            preferred_provider=preferred_provider,
            operation=f"rewrite_{persona}",
            temperature=temperature,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        answer_text = str(data.get("answer") or "").strip()
        metrics = cls.calculate_metrics(answer_text)

        return HumanizedAnswer(
            persona=persona,
            persona_name=persona_info["name"],
            answer_text=answer_text,
            metrics=metrics,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )

    @classmethod
    def generate_all_personas(
        cls,
        question: str,
        context: Optional[str] = None,
        original_ai_answer: Optional[str] = None,
        personas: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None,
        temperature: float = 0.75,
    ) -> MultiPersonaResult:
        """
        Generates answers for multiple personas simultaneously in one LLM call for high efficiency and speed.
        """
        target_personas = personas or list(PERSONA_PROMPTS.keys())

        persona_descriptions = "\n".join(
            f'- Key "{p}": ({PERSONA_PROMPTS[p]["name"]}) - {PERSONA_PROMPTS[p]["description"]}'
            for p in target_personas
            if p in PERSONA_PROMPTS
        )

        prompt = f"""Bạn là chuyên gia mô phỏng ngôn ngữ hội thoại phỏng vấn tiếng Việt.
Nhiệm vụ: Tạo các phiên bản câu trả lời phỏng vấn theo phong cách NÓI THẬT CỦA CON NGƯỜI (Spoken Natural Human Responses), ĐẶC BIỆT KHÔNG ĐƯỢC GIỐNG VĂN VIẾT CỦA AI.

CÂU HỎI PHỎNG VẤN:
"{question}"

{f'BỐI CẢNH ỨNG VIÊN / CV / JD: {context}' if context else ''}
{f'CÂU TRẢ LỜI AI GỐC CẦN THAM KHẢO Ý: {original_ai_answer}' if original_ai_answer else ''}

Hãy sinh ra câu trả lời cho các phong cách người thật sau:
{persona_descriptions}

YÊU CẦU BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG dùng gạch đầu dòng, danh sách 1 2 3, cấu trúc bài luận hay văn phong sách giáo khoa của AI.
2. 100% văn nói tiếng Việt (Spoken Vietnamese), có từ đệm ("ừm...", "dạ...", "thì..."), có ngập ngừng suy nghĩ ("..."), có cảm xúc thật.
3. Mỗi phong cách phải có nét riêng biệt rõ ràng như đã mô tả.
4. Trả về đúng định dạng JSON sau:
{{
{', '.join(f'  "{p}": "nội dung câu trả lời cho {p}"' for p in target_personas)}
}}
"""
        started = time.perf_counter()
        data, provider, model, _ = cls._execute_llm_json(
            prompt,
            preferred_provider=preferred_provider,
            operation="generate_multi_humanized",
            temperature=temperature,
        )
        total_latency_ms = round((time.perf_counter() - started) * 1000)

        result = MultiPersonaResult(
            question_text=question,
            context_text=context,
            original_ai_answer=original_ai_answer,
        )

        for p in target_personas:
            ans_text = str(data.get(p) or "").strip()
            if ans_text:
                result.answers[p] = HumanizedAnswer(
                    persona=p,
                    persona_name=PERSONA_PROMPTS.get(p, {}).get("name", p),
                    answer_text=ans_text,
                    metrics=cls.calculate_metrics(ans_text),
                    provider=provider,
                    model=model,
                    latency_ms=total_latency_ms,
                )

        return result

