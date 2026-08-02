"""
Адаптер над обраною демо-системою (System Under Test).

Дає ОДИН уніфікований метод `generate(case)` для всіх треків, щоб крок генерації
(`src/generate.py`) був однаковим. Стек і патерн дзеркалять ДЗ 8/9:
генеруємо ОДИН раз -> зберігаємо у outputs/generations.json -> оцінюємо детерміновано офлайн.

⚠️ Базове підключення системи дане. ТВОЯ робота — за потреби розширити його
(multi-turn послідовності, reset між кейсами, ML-предикти) і написати тести/метрики.
AI-асистент пояснює й рев'ює, але не пише твої метрики (див. CLAUDE.md / AGENTS.md).
"""

from __future__ import annotations

import os

# Обери трек: 'A' = chatbot, 'B' = rag, 'C' = ml, 'D' = agent
TRACK = os.getenv("TRACK", "B")


class StudentSUT:
    def __init__(self) -> None:
        if TRACK == "A":
            from chatbot_sut import ChatbotSUT
            self._bot = ChatbotSUT()
        elif TRACK == "B":
            from rag_sut import RagSUT
            self._rag = RagSUT()
        elif TRACK == "C":
            import joblib
            from pathlib import Path
            bundle = joblib.load(Path(__file__).resolve().parents[1] / "sut_artifact.joblib")
            self._pipe = bundle["pipeline"]
            self._features = bundle["features"]
        elif TRACK == "D":
            from agent_sut import AgentSUT
            self._agent = AgentSUT()
        else:
            raise ValueError("Unknown TRACK: " + str(TRACK))

    def generate(self, case: dict) -> dict:
        """Повертає dict із полем 'output' (+ 'sources' для RAG)."""
        if TRACK == "A":
            # Базовий single-turn. TODO: для multi-turn кейсів подавай послідовність кроків
            # і викликай self._bot.reset() між незалежними кейсами.
            self._bot.reset()
            return {"output": self._bot.chat(case["input"])}

        if TRACK == "B":
            result = self._rag.ask(case["input"])
            return {"output": result["answer"], "sources": result["sources"]}

        if TRACK == "C":
            # Для ML зазвичай зручніше зберігати предикти по рядках production_data.csv.
            # TODO: за потреби адаптуй під свій eval-набір.
            import pandas as pd
            row = pd.DataFrame([case["input"]]) if isinstance(case["input"], dict) else None
            if row is None:
                raise NotImplementedError("Для треку C збережи предикти по production_data.csv")
            return {"output": int(self._pipe.predict(row[self._features])[0])}

        if TRACK == "D":
            # Агент: зберігаємо повну трасу (для tool call accuracy, trajectory, red-team).
            tr = self._agent.handle(case["input"])
            return {
                "output": tr["output"],
                "selected_agent": tr["selected_agent"],
                "tool_calls": tr["tool_calls"],
            }

        raise ValueError("Unknown TRACK")
