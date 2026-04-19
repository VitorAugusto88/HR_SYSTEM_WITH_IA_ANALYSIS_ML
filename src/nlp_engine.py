"""
Motor de NLP para geração de perguntas de entrevista e análise de respostas.
Usa a API da Anthropic (Claude Haiku para geração, Sonnet para análise).
Claude Haiku é mais rápido/barato; Claude Sonnet oferece maior precisão analítica.
"""

import json
import anthropic
from src.config import COMPANY_CULTURE, AVAILABLE_POSITIONS

# Modelos disponíveis (troque conforme necessidade de velocidade x qualidade)
MODEL_FAST     = "claude-haiku-4-5-20251001"   # rápido, ideal para geração de perguntas
MODEL_ANALYSIS = "claude-sonnet-4-6"            # mais preciso, ideal para avaliar respostas


def get_client(api_key: str) -> anthropic.Anthropic:
    """Retorna um cliente Anthropic autenticado."""
    return anthropic.Anthropic(api_key=api_key)


def _call(client: anthropic.Anthropic, system: str, user: str,
          model: str = MODEL_FAST, max_tokens: int = 3000, temperature: float = 0.7) -> str:
    """Helper interno: chama a API e retorna o texto da resposta."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


def _parse_json(raw: str) -> any:
    """Remove cerca de markdown se presente e faz parse do JSON."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


def generate_interview_questions(client: anthropic.Anthropic,
                                  position_key: str,
                                  num_questions: int = 8) -> list[dict]:
    """Gera perguntas de entrevista personalizadas para a vaga e cultura da empresa."""
    position = AVAILABLE_POSITIONS[position_key]

    system = "Você é um especialista em RH. Responda APENAS com JSON válido, sem markdown, sem explicações."

    user = f"""Você é recrutador sênior da empresa {COMPANY_CULTURE['nome']}.

CULTURA DA EMPRESA:
- Missão: {COMPANY_CULTURE['missao']}
- Valores: {', '.join(COMPANY_CULTURE['valores'])}
- Ambiente: {COMPANY_CULTURE['ambiente']}
- Estilo de liderança: {COMPANY_CULTURE['estilo_lideranca']}
- Perfil ideal: {json.dumps(COMPANY_CULTURE['perfil_ideal'], ensure_ascii=False)}

VAGA: {position['titulo']}
- Departamento: {position['departamento']}
- Nível: {position['nivel']}
- Habilidades técnicas: {', '.join(position['habilidades_tecnicas'])}
- Habilidades comportamentais: {', '.join(position['habilidades_comportamentais'])}
- Experiência mínima: {position['experiencia_minima_anos']} anos

Gere exatamente {num_questions} perguntas de entrevista. Distribua: 2 técnicas, 2 comportamentais, 2 de cultura, 2 situacionais.

Retorne APENAS um JSON válido no formato:
[
  {{
    "pergunta": "...",
    "categoria": "tecnica|comportamental|cultura|situacional",
    "avalia": ["competencia1", "competencia2"],
    "criterios_boa_resposta": "..."
  }}
]"""

    raw = _call(client, system, user, model=MODEL_FAST, max_tokens=3000, temperature=0.7)
    return _parse_json(raw)


def analyze_candidate_responses(client: anthropic.Anthropic,
                                 position_key: str,
                                 questions_and_answers: list[dict]) -> dict:
    """
    Analisa as respostas do candidato e retorna scores de 1-10 para cada dimensão.
    Usa Claude Sonnet para maior precisão analítica.
    """
    position = AVAILABLE_POSITIONS[position_key]

    qa_text = ""
    for i, qa in enumerate(questions_and_answers, 1):
        qa_text += f"\nPergunta {i} [{qa['categoria']}]: {qa['pergunta']}\n"
        qa_text += f"Resposta: {qa['resposta']}\n"

    system = "Você é um avaliador de RH experiente e criterioso. Responda APENAS com JSON válido, sem markdown."

    user = f"""Você avalia candidatos para a empresa {COMPANY_CULTURE['nome']}.

CULTURA DA EMPRESA:
- Valores: {', '.join(COMPANY_CULTURE['valores'])}
- Perfil ideal: {json.dumps(COMPANY_CULTURE['perfil_ideal'], ensure_ascii=False)}

VAGA: {position['titulo']} ({position['nivel']})
Departamento: {position['departamento']}

ENTREVISTA DO CANDIDATO:
{qa_text}

Avalie o candidato em cada dimensão com scores de 1.0 a 10.0. Seja criterioso e realista.

Retorne APENAS o JSON abaixo (sem nenhum texto fora dele):
{{
  "scores": {{
    "soft_skills": <float 1-10>,
    "technical_skills": <float 1-10>,
    "leadership": <float 1-10>,
    "communication": <float 1-10>,
    "adaptability": <float 1-10>,
    "innovation": <float 1-10>,
    "teamwork": <float 1-10>,
    "culture_fit": <float 1-10>,
    "job_satisfaction_potential": <int 1-4>,
    "work_life_balance_expectation": <int 1-4>,
    "engagement_potential": <float 1-10>,
    "growth_potential": <float 1-10>
  }},
  "resumo_avaliacao": "...",
  "pontos_fortes": ["...", "..."],
  "pontos_atencao": ["...", "..."],
  "recomendacao": "APROVADO|APROVADO_COM_RESSALVAS|REPROVADO",
  "justificativa_recomendacao": "..."
}}"""

    raw = _call(client, system, user, model=MODEL_ANALYSIS, max_tokens=2000, temperature=0.3)
    return _parse_json(raw)


def generate_followup_question(client: anthropic.Anthropic,
                                position_key: str,
                                previous_qa: list[dict],
                                area_to_explore: str) -> str:
    """Gera uma pergunta de follow-up contextualizada nas respostas anteriores."""
    position = AVAILABLE_POSITIONS[position_key]

    qa_text = ""
    for qa in previous_qa[-3:]:  # Últimas 3 interações para contexto
        qa_text += f"P: {qa['pergunta']}\nR: {qa['resposta']}\n\n"

    system = "Você é um entrevistador de RH experiente. Responda APENAS com a pergunta, sem formatação extra."

    user = f"""Entrevista em andamento para a vaga de {position['titulo']}.

Trecho da conversa:
{qa_text}

Gere UMA pergunta de follow-up natural que aprofunde a área: {area_to_explore}
A pergunta deve soar como parte fluida da conversa, sem ser mecânica."""

    return _call(client, system, user, model=MODEL_FAST, max_tokens=300, temperature=0.7)
