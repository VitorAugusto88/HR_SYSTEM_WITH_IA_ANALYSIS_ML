"""
Interface de RH com IA — TechNova Solutions  (versão melhorada)
Melhorias: comparação de candidatos, follow-up dinâmico, explicação de fatores,
           exportação de relatório, mais vagas, UX aprimorada.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os, json, io
from datetime import datetime
from io import BytesIO

sys.path.insert(0, os.path.dirname(__file__))

from src.config import COMPANY_CULTURE, AVAILABLE_POSITIONS
from src.ml_predictor import HRPredictor, compare_candidates

GREEN = "#059669"
GREEN_DARK = "#064e3b"
GREEN_LIGHT = "#d1fae5"
RED = "#dc2626"
DARK = "#111827"
GRAY = "#6b7280"
BLUE = "#2563eb"

st.set_page_config(
    page_title="TechNova HR",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .stApp { background:#fff; }
  .stApp,.stApp p,.stApp span,.stApp label,.stApp li,.stApp div { color:#111827 !important; }
  h1,h2,h3,h4 { color:#111827 !important; font-weight:700 !important; }
  section[data-testid="stSidebar"] { background:#f9fafb; border-right:1px solid #e5e7eb; }
  section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color:#111827 !important; }
  .stButton>button { background:#059669 !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
  .stButton>button:hover { background:#047857 !important; }
  .stFormSubmitButton>button { background:#059669 !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
  div[data-testid="stMetric"] { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:.8rem; }
  div[data-testid="stMetric"] label { color:#6b7280 !important; font-size:.85rem !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color:#111827 !important; font-weight:700 !important; }
  .stProgress>div>div { background:#059669 !important; }
  .header-bar { background:#059669; padding:2rem; border-radius:12px; margin-bottom:1.5rem; }
  .header-bar h1 { color:#fff !important; font-size:1.8rem; margin:0; }
  .header-bar p { color:rgba(255,255,255,.9) !important; margin:.3rem 0 0; font-size:1rem; }
  .q-card { background:#f9fafb; border:1px solid #e5e7eb; border-left:4px solid #059669; padding:1.2rem 1.5rem; border-radius:0 10px 10px 0; margin:1rem 0; }
  .q-label { font-size:.78rem; font-weight:700; color:#059669 !important; text-transform:uppercase; letter-spacing:.04em; }
  .q-tag { display:inline-block; background:#d1fae5; color:#064e3b !important; font-size:.72rem; font-weight:600; padding:2px 10px; border-radius:12px; margin-left:8px; }
  .q-followup { display:inline-block; background:#dbeafe; color:#1e40af !important; font-size:.72rem; font-weight:600; padding:2px 10px; border-radius:12px; margin-left:8px; }
  .q-text { font-size:1.05rem; color:#111827 !important; line-height:1.6; margin-top:.5rem; }
  .q-eval { font-size:.8rem; color:#6b7280 !important; margin-top:.4rem; }
  .step { background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:1.2rem; text-align:center; }
  .step .num { display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:50%; background:#059669; color:#fff !important; font-weight:700; font-size:.9rem; margin-bottom:.6rem; }
  .step h4 { color:#111827 !important; font-size:.95rem; margin:.4rem 0; }
  .step p { color:#6b7280 !important; font-size:.85rem; line-height:1.5; }
  .cand-badge { background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:.6rem 1rem; font-size:.9rem; color:#111827 !important; display:inline-block; }
  .rec-ok   { background:#d1fae5; border:1px solid #059669; border-radius:10px; padding:1rem 1.2rem; color:#064e3b !important; }
  .rec-warn { background:#fef3c7; border:1px solid #d97706; border-radius:10px; padding:1rem 1.2rem; color:#92400e !important; }
  .rec-fail { background:#fee2e2; border:1px solid #dc2626; border-radius:10px; padding:1rem 1.2rem; color:#991b1b !important; }
  .factor-bar { height:8px; border-radius:4px; background:#e5e7eb; margin-top:4px; }
  .factor-fill { height:8px; border-radius:4px; background:#059669; }
  .rank-gold   { color:#d97706 !important; font-weight:700; }
  .rank-silver { color:#6b7280 !important; font-weight:700; }
  .rank-bronze { color:#b45309 !important; font-weight:700; }
  .stTextArea textarea { border:1px solid #d1d5db; border-radius:8px; color:#111827 !important; }
  .stTextArea textarea:focus { border-color:#059669; box-shadow:0 0 0 2px rgba(5,150,105,.12); }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    defaults = {
        "page": "home",
        "selected_position": None,
        "questions": [],
        "answers": {},
        "current_question_idx": 0,
        "interview_complete": False,
        "nlp_analysis": None,
        "predictions": None,
        "candidate_info": {},
        "api_key_set": False,
        "followup_mode": False,
        "followup_question": None,
        "all_candidates": [],  # lista de candidatos finalizados para comparação
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 TechNova HR")
        st.caption("Recrutamento com IA & ML")
        st.divider()

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        if api_key:
            st.session_state["api_key"] = api_key
            st.session_state["api_key_set"] = True
            st.success("✅ API Key configurada")
        else:
            st.warning("⚠️ Insira sua Anthropic API Key")

        st.divider()
        pages = {
            "home": "🏠 Início",
            "culture": "🏢 Cultura",
            "interview": "🎙️ Nova Entrevista",
            "compare": "📊 Comparar Candidatos",
            "dashboard": "📈 Dashboard Analytics",
        }
        for key, label in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state["page"] = key
                st.rerun()

        if st.session_state["all_candidates"]:
            st.divider()
            st.caption(
                f"👥 {len(st.session_state['all_candidates'])} candidato(s) avaliado(s)"
            )


def render_home():
    st.markdown(
        '<div class="header-bar"><h1>🧠 TechNova HR — Sistema de Recrutamento com IA</h1>'
        "<p>Machine Learning + NLP para seleção mais inteligente e justa</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vagas Abertas", len(AVAILABLE_POSITIONS))
    c2.metric("Modelo Base", "Notebook 02")
    c3.metric("NLP Engine", "Claude API")
    c4.metric("Clusters", "K dinâmico")
    c5.metric("Candidatos", len(st.session_state["all_candidates"]))

    st.divider()
    st.markdown("### Como funciona")
    cols = st.columns(4)
    steps = [
        ("1", "Selecione a Vaga", "Pesos de avaliação personalizados por cargo."),
        ("2", "Entreviste", "IA gera perguntas + follow-ups dinâmicos."),
        (
            "3",
            "Analise os Resultados",
            "ML prevê compatibilidade, fit e risco de saída.",
        ),
        ("4", "Compare Candidatos", "Ranking automático entre todos os entrevistados."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(
            f'<div class="step"><div class="num">{num}</div><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Vagas Disponíveis")
    for key, pos in AVAILABLE_POSITIONS.items():
        with st.expander(
            f"**{pos['titulo']}** — {pos['departamento']} | {pos['nivel']} | {pos['faixa_salarial']}"
        ):
            st.markdown(f"_{pos.get('descricao', '')}_")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Técnicas:** {', '.join(pos['habilidades_tecnicas'])}")
            c2.markdown(
                f"**Comportamentais:** {', '.join(pos['habilidades_comportamentais'])}"
            )
            if st.button("🎙️ Iniciar Entrevista", key=f"start_{key}"):
                st.session_state.update(
                    selected_position=key,
                    page="interview",
                    questions=[],
                    answers={},
                    current_question_idx=0,
                    interview_complete=False,
                    nlp_analysis=None,
                    predictions=None,
                    candidate_info={},
                    followup_mode=False,
                    followup_question=None,
                )
                st.rerun()


def render_culture():
    st.markdown("## 🏢 Cultura da TechNova Solutions")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Missão")
        st.info(COMPANY_CULTURE["missao"])
        st.markdown("### Valores")
        for v in COMPANY_CULTURE["valores"]:
            st.markdown(f"- {v}")
        st.markdown(
            f"### Ambiente\n**Modelo:** {COMPANY_CULTURE['ambiente']}  \n"
            f"**Liderança:** {COMPANY_CULTURE['estilo_lideranca']}  \n"
            f"**Comunicação:** {COMPANY_CULTURE['tom_comunicacao']}"
        )
    with c2:
        st.markdown("### Perfil Ideal")
        for k, v in COMPANY_CULTURE["perfil_ideal"].items():
            st.markdown(f"**{k.replace('_',' ').title()}:** {v}")
        st.markdown("### Benefícios")
        for b in COMPANY_CULTURE["beneficios"]:
            st.markdown(f"- {b}")

    cats = [
        "Inovação",
        "Colaboração",
        "Transparência",
        "Diversidade",
        "Equilíbrio",
        "Aprendizado",
        "Resultados",
    ]
    vals = [9, 9, 8, 9, 8, 9, 8]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(5,150,105,.12)",
            line=dict(color=GREEN, width=2),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10]), bgcolor="white"),
        height=400,
        paper_bgcolor="white",
        font=dict(color=DARK),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_interview():
    if not st.session_state.get("api_key_set"):
        st.error("⚠️ Configure sua Anthropic API Key na sidebar.")
        return

    pos_key = st.session_state.get("selected_position")
    if not pos_key:
        st.warning("Selecione uma vaga na página inicial.")
        if st.button("Ir para Início"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    pos = AVAILABLE_POSITIONS[pos_key]
    st.markdown(f"## 🎙️ Entrevista: {pos['titulo']}")
    st.markdown(
        f"**{pos['departamento']}** | **{pos['nivel']}** | {pos['faixa_salarial']}"
    )
    st.divider()

    # Dados do candidato
    if not st.session_state.get("candidate_info", {}).get("nome"):
        st.markdown("### 👤 Dados do Candidato")
        with st.form("cand_form"):
            c1, c2 = st.columns(2)
            nome = c1.text_input("Nome completo")
            idade = c1.number_input("Idade", 18, 65, 30)
            exp = c2.number_input("Anos de Experiência", 0, 40, 3)
            emp = c2.number_input("Empresas Anteriores", 0, 15, 2)
            escol_opt = [
                "Ensino Médio",
                "Graduação",
                "Pós-Graduação",
                "Mestrado",
                "Doutorado",
            ]
            escol = c1.selectbox("Escolaridade", escol_opt, index=1)
            escol_map = {e: i + 1 for i, e in enumerate(escol_opt)}
            if st.form_submit_button("✅ Continuar"):
                if not nome.strip():
                    st.error("Informe o nome do candidato.")
                else:
                    st.session_state["candidate_info"] = {
                        "nome": nome.strip(),
                        "idade": idade,
                        "experiencia": exp,
                        "empresas": emp,
                        "escolaridade": escol_map[escol],
                        "escolaridade_str": escol,
                    }
                    st.rerun()
        return

    cand = st.session_state["candidate_info"]
    st.markdown(
        f'<div class="cand-badge">👤 <b>{cand["nome"]}</b> &mdash; '
        f'{cand["idade"]} anos | {cand["experiencia"]} anos exp. | {cand.get("escolaridade_str","")} | '
        f'{cand["empresas"]} empresa(s)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Gerar perguntas iniciais
    if not st.session_state["questions"]:
        with st.spinner("🤖 Gerando perguntas personalizadas com IA..."):
            from src.nlp_engine import get_client, generate_interview_questions

            client = get_client(st.session_state["api_key"])
            st.session_state["questions"] = generate_interview_questions(
                client, pos_key, num_questions=8
            )
            st.rerun()

    questions = st.session_state["questions"]
    total = len(questions)
    answered = len(st.session_state["answers"])
    st.progress(answered / total, text=f"{answered}/{total} perguntas respondidas")

    if not st.session_state["interview_complete"]:

        # ——— Follow-up dinâmico ———
        if st.session_state.get("followup_mode") and st.session_state.get(
            "followup_question"
        ):
            fq = st.session_state["followup_question"]
            st.markdown(
                f'<div class="q-card">'
                f'<span class="q-label">Follow-up</span>'
                f'<span class="q-followup">🔍 Aprofundamento</span>'
                f'<div class="q-text">{fq}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.form("followup_ans"):
                resp = st.text_area(
                    "Resposta do candidato:",
                    height=120,
                    placeholder="Registre a resposta...",
                )
                c1, c2 = st.columns(2)
                ok = c1.form_submit_button("➡️ Continuar", use_container_width=True)
                skip = c2.form_submit_button("⏭️ Pular", use_container_width=True)
                if ok and resp.strip():
                    idx = st.session_state["current_question_idx"] - 1
                    prev = st.session_state["answers"].get(idx, "")
                    st.session_state["answers"][idx] = (
                        prev + f"\n[Follow-up]: {resp.strip()}"
                    )
                    st.session_state["followup_mode"] = False
                    st.session_state["followup_question"] = None
                    st.rerun()
                elif skip:
                    st.session_state["followup_mode"] = False
                    st.session_state["followup_question"] = None
                    st.rerun()
        else:
            idx = st.session_state["current_question_idx"]
            if idx < total:
                q = questions[idx]
                st.markdown(
                    f'<div class="q-card">'
                    f'<span class="q-label">Pergunta {idx+1} de {total}</span>'
                    f'<span class="q-tag">{q["categoria"]}</span>'
                    f'<div class="q-text">{q["pergunta"]}</div>'
                    f'<div class="q-eval">Avalia: {", ".join(q["avalia"])}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                with st.form(f"ans_{idx}"):
                    resp = st.text_area(
                        "Resposta do candidato:",
                        height=150,
                        placeholder="Digite a resposta do candidato...",
                    )
                    c1, c2, c3 = st.columns(3)
                    ok = c1.form_submit_button("➡️ Próxima", use_container_width=True)
                    do_fup = c2.form_submit_button(
                        "🔍 Follow-up", use_container_width=True
                    )
                    skip = c3.form_submit_button("⏭️ Pular", use_container_width=True)

                    if (ok or do_fup) and resp.strip():
                        st.session_state["answers"][idx] = resp.strip()
                        st.session_state["current_question_idx"] = idx + 1
                        if idx + 1 >= total:
                            st.session_state["interview_complete"] = True
                        elif do_fup:
                            # Gerar follow-up assíncrono na próxima run
                            st.session_state["followup_mode"] = True
                            st.session_state["followup_pending"] = {
                                "position_key": pos_key,
                                "last_qa": {
                                    "pergunta": q["pergunta"],
                                    "resposta": resp.strip(),
                                    "categoria": q["categoria"],
                                },
                            }
                        st.rerun()
                    elif skip:
                        st.session_state["answers"][idx] = "(Pulada)"
                        st.session_state["current_question_idx"] = idx + 1
                        if idx + 1 >= total:
                            st.session_state["interview_complete"] = True
                        st.rerun()

        # Gerar follow-up se pendente
        if st.session_state.get("followup_pending"):
            pending = st.session_state.pop("followup_pending")
            with st.spinner("🤖 Gerando pergunta de follow-up..."):
                from src.nlp_engine import get_client, generate_followup_question

                client = get_client(st.session_state["api_key"])
                prev_qa = [
                    {
                        "pergunta": pending["last_qa"]["pergunta"],
                        "resposta": pending["last_qa"]["resposta"],
                    }
                ]
                fq = generate_followup_question(
                    client,
                    pending["position_key"],
                    prev_qa,
                    pending["last_qa"]["categoria"],
                )
                st.session_state["followup_question"] = fq
            st.rerun()

        # Histórico expansível
        if answered > 0:
            with st.expander(f"📋 Respostas anteriores ({answered})"):
                for i in range(answered):
                    st.markdown(
                        f"**P{i+1} [{questions[i]['categoria']}]:** {questions[i]['pergunta']}"
                    )
                    st.markdown(f"↳ {st.session_state['answers'].get(i, '')}")
                    st.divider()

        if answered >= 4 and not st.session_state["interview_complete"]:
            if st.button("⏹️ Finalizar Antecipadamente"):
                st.session_state["interview_complete"] = True
                st.rerun()
    else:
        if st.session_state["nlp_analysis"] is None:
            with st.spinner("🧠 Analisando respostas com IA..."):
                from src.nlp_engine import get_client, analyze_candidate_responses

                client = get_client(st.session_state["api_key"])
                qa = [
                    {
                        "pergunta": q["pergunta"],
                        "categoria": q["categoria"],
                        "resposta": st.session_state["answers"][i],
                    }
                    for i, q in enumerate(questions)
                    if i in st.session_state["answers"]
                ]
                nlp_result = analyze_candidate_responses(client, pos_key, qa)
                st.session_state["nlp_analysis"] = nlp_result

                predictor = HRPredictor()
                st.session_state["predictions"] = predictor.full_prediction(
                    nlp_result, pos, cand, pos_key
                )

                # Guardar candidato na lista de comparação
                entry = {
                    "nome": cand["nome"],
                    "vaga": pos["titulo"],
                    "pos_key": pos_key,
                    "nlp_analysis": nlp_result,
                    "predictions": st.session_state["predictions"],
                    "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                }
                # Atualizar se já existir (re-entrevista)
                existing = [
                    c
                    for c in st.session_state["all_candidates"]
                    if c["nome"] == cand["nome"] and c["pos_key"] == pos_key
                ]
                if existing:
                    idx_e = st.session_state["all_candidates"].index(existing[0])
                    st.session_state["all_candidates"][idx_e] = entry
                else:
                    st.session_state["all_candidates"].append(entry)
            st.rerun()
        render_results()


def render_results():
    nlp = st.session_state["nlp_analysis"]
    pred = st.session_state["predictions"]
    cand = st.session_state["candidate_info"]
    pos_key = st.session_state["selected_position"]
    pos = AVAILABLE_POSITIONS[pos_key]

    st.markdown(f"## ✅ Resultado: {cand['nome']}")
    st.markdown(f"**{pos['titulo']}** | {pos['departamento']}")
    st.divider()

    # Recomendação
    rec = nlp.get("recomendacao", "N/A")
    css = {
        "APROVADO": "rec-ok",
        "APROVADO_COM_RESSALVAS": "rec-warn",
        "REPROVADO": "rec-fail",
    }.get(rec, "rec-warn")
    icons = {"APROVADO": "✅", "APROVADO_COM_RESSALVAS": "⚠️", "REPROVADO": "❌"}
    st.markdown(
        f'<div class="{css}"><b>{icons.get(rec,"")} Recomendação: {rec.replace("_"," ")}</b>'
        f'<br>{nlp.get("justificativa_recomendacao","")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Métricas
    compat = pred["compatibility"]
    attr = pred["attrition"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Compat. Vaga",
        f"{compat['compatibilidade_vaga']:.0f}%",
        delta="Aprovado" if compat["compatibilidade_vaga"] >= 70 else "Abaixo da meta",
    )
    c2.metric(
        "Fit Cultural",
        f"{compat['compatibilidade_cultura']:.0f}%",
        delta="Aprovado" if compat["compatibilidade_cultura"] >= 65 else "Atenção",
    )
    c3.metric(
        "Risco de Saída",
        f"{attr['probabilidade_sair']:.0%}",
        delta=f"Risco {attr['risco']}",
        delta_color="inverse",
    )
    c4.metric(
        "Potencial",
        f"{compat['potencial_crescimento']:.0f}%",
        delta="Alto" if compat["potencial_crescimento"] >= 75 else "Regular",
    )
    st.caption(
        f"Threshold do modelo de attrition: {attr.get('threshold_modelo', 0.5):.3f} | "
        f"Predicao operacional: {'maior risco de saida' if attr.get('predicao_binaria') else 'maior chance de permanencia'}"
    )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        sc = nlp["scores"]
        cats = [
            "Soft Skills",
            "Técnico",
            "Liderança",
            "Comunicação",
            "Adaptabilidade",
            "Inovação",
            "Equipe",
            "Fit Cultural",
        ]
        vals = [
            sc.get(k, 5)
            for k in [
                "soft_skills",
                "technical_skills",
                "leadership",
                "communication",
                "adaptability",
                "innovation",
                "teamwork",
                "culture_fit",
            ]
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(5,150,105,.12)",
                line=dict(color=GREEN, width=2),
                name="Candidato",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=[7] * 9,
                theta=cats + [cats[0]],
                line=dict(color="#d1d5db", width=1, dash="dash"),
                name="Meta 7.0",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10]), bgcolor="white"),
            height=400,
            paper_bgcolor="white",
            font=dict(color=DARK),
            legend=dict(font=dict(size=11)),
            title="Radar de Competências",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "indicator"}] * 2] * 2)
        for row, col, val, title, bar_color in [
            (1, 1, compat["compatibilidade_vaga"], "Compat. Vaga", GREEN),
            (1, 2, compat["compatibilidade_cultura"], "Fit Cultural", "#10b981"),
            (2, 1, attr["probabilidade_sair"] * 100, "Risco Saída", RED),
            (2, 2, compat["potencial_crescimento"], "Potencial", GREEN_DARK),
        ]:
            is_risk = title == "Risco Saída"
            steps = (
                [
                    {"range": [0, 25], "color": "#d1fae5"},
                    {"range": [25, 40], "color": "#fef9c3"},
                    {"range": [40, 100], "color": "#fecaca"},
                ]
                if is_risk
                else [
                    {"range": [0, 50], "color": "#fecaca"},
                    {"range": [50, 70], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#d1fae5"},
                ]
            )
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": title, "font": {"size": 13, "color": DARK}},
                    number={"font": {"color": RED if is_risk else GREEN_DARK}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": bar_color},
                        "bgcolor": "white",
                        "steps": steps,
                    },
                ),
                row=row,
                col=col,
            )
        fig.update_layout(
            height=400,
            paper_bgcolor="white",
            font=dict(color=DARK),
            title="Gauges de Desempenho",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ——— Seção nova: Explicação dos Fatores ———
    st.divider()
    st.markdown("### 🔍 Fatores que Mais Influenciaram a Compatibilidade com a Vaga")
    explanation = pred.get("explanation", {})
    contributions = explanation.get("contributions", [])
    if contributions:
        cols = st.columns(len(contributions))
        for col, factor in zip(cols, contributions):
            score = factor["score"]
            color = GREEN if score >= 7 else ("#f59e0b" if score >= 5 else RED)
            pct = int(score * 10)
            col.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:.75rem;color:#6b7280;font-weight:600'>{factor['fator']}</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{color}'>{score}</div>"
                f"<div style='font-size:.7rem;color:#9ca3af'>peso {factor['peso']:.0%}</div>"
                f"<div class='factor-bar'><div class='factor-fill' style='width:{pct}%;background:{color}'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            f"💪 Maior força: **{explanation.get('top_strength','N/A')}** | "
            f"⚠️ Ponto crítico: **{explanation.get('top_weakness','N/A')}**"
        )

    st.divider()
    c1, c2 = st.columns(2)
    cluster = pred["cluster"]
    with c1:
        st.markdown("### 🎯 Perfil (Clustering K-Means)")
        st.markdown(f"**Cluster:** {cluster['perfil']} (#{cluster['cluster']})")
        st.markdown(f"**Fit Score:** {cluster['fit_score']:.2f}")
        for perfil, dist in cluster["distancias"].items():
            st.progress(max(0.0, min(1.0, 1 - dist / 10)), text=f"{perfil}: {dist:.2f}")

    with c2:
        st.markdown("### 📝 Avaliação NLP")
        st.markdown(nlp.get("resumo_avaliacao", ""))
        st.markdown("**Pontos Fortes:**")
        for p in nlp.get("pontos_fortes", []):
            st.markdown(f"✅ {p}")
        st.markdown("**Pontos de Atenção:**")
        for p in nlp.get("pontos_atencao", []):
            st.markdown(f"⚠️ {p}")

    # ——— Exportar relatório ———
    st.divider()
    st.markdown("### 📄 Exportar Relatório")
    report_text = generate_text_report(cand, pos, nlp, pred)
    st.download_button(
        label="⬇️ Baixar Relatório (.txt)",
        data=report_text,
        file_name=f"relatorio_{cand['nome'].replace(' ','_')}_{pos_key}.txt",
        mime="text/plain",
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    if c1.button("🎙️ Nova Entrevista", use_container_width=True):
        st.session_state.update(
            selected_position=None,
            questions=[],
            answers={},
            current_question_idx=0,
            interview_complete=False,
            nlp_analysis=None,
            predictions=None,
            candidate_info={},
            page="home",
            followup_mode=False,
            followup_question=None,
        )
        st.rerun()
    if c2.button("📊 Comparar Candidatos", use_container_width=True):
        st.session_state["page"] = "compare"
        st.rerun()
    if c3.button("📈 Dashboard", use_container_width=True):
        st.session_state["page"] = "dashboard"
        st.rerun()


def generate_text_report(cand, pos, nlp, pred) -> str:
    """Gera um relatório textual estruturado do candidato."""
    compat = pred["compatibility"]
    attr = pred["attrition"]
    cluster = pred["cluster"]
    exp = pred.get("explanation", {})
    lines = [
        "=" * 60,
        "RELATÓRIO DE ENTREVISTA — TECHNOVA HR",
        "=" * 60,
        f"Data:       {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"Candidato:  {cand['nome']}",
        f"Idade:      {cand['idade']} anos",
        f"Experiência: {cand['experiencia']} anos ({cand['empresas']} empresa(s))",
        f"Escolaridade: {cand.get('escolaridade_str','')}",
        "",
        f"Vaga:       {pos['titulo']}",
        f"Departamento: {pos['departamento']} | Nível: {pos['nivel']}",
        f"Faixa Sal:  {pos['faixa_salarial']}",
        "",
        "─" * 60,
        "RECOMENDAÇÃO",
        "─" * 60,
        f"Resultado: {nlp.get('recomendacao','N/A')}",
        f"Justificativa: {nlp.get('justificativa_recomendacao','')}",
        "",
        "─" * 60,
        "SCORES DE COMPATIBILIDADE",
        "─" * 60,
        f"Compatibilidade com a Vaga: {compat['compatibilidade_vaga']:.1f}%",
        f"Fit Cultural:               {compat['compatibilidade_cultura']:.1f}%",
        f"Potencial de Crescimento:   {compat['potencial_crescimento']:.1f}%",
        f"Risco de Saída:             {attr['probabilidade_sair']:.1%} ({attr['risco']})",
        f"Perfil Cluster:             {cluster['perfil']}",
        "",
        "─" * 60,
        "COMPETÊNCIAS AVALIADAS (NLP)",
        "─" * 60,
    ]
    sc = nlp.get("scores", {})
    label_map = {
        "soft_skills": "Soft Skills",
        "technical_skills": "Técnico",
        "leadership": "Liderança",
        "communication": "Comunicação",
        "adaptability": "Adaptabilidade",
        "innovation": "Inovação",
        "teamwork": "Trabalho em Equipe",
        "culture_fit": "Fit Cultural",
        "engagement_potential": "Engajamento",
        "growth_potential": "Potencial de Crescimento",
    }
    for k, label in label_map.items():
        if k in sc:
            lines.append(f"  {label:30s}: {sc[k]:.1f}/10")

    lines += [
        "",
        "─" * 60,
        "PONTOS FORTES",
        "─" * 60,
    ]
    for p in nlp.get("pontos_fortes", []):
        lines.append(f"  + {p}")

    lines += [
        "",
        "─" * 60,
        "PONTOS DE ATENÇÃO",
        "─" * 60,
    ]
    for p in nlp.get("pontos_atencao", []):
        lines.append(f"  ! {p}")

    lines += [
        "",
        "─" * 60,
        "RESUMO DA AVALIAÇÃO",
        "─" * 60,
        nlp.get("resumo_avaliacao", ""),
        "",
        "─" * 60,
        "FATOR DE MAIOR IMPACTO NA COMPATIBILIDADE",
        "─" * 60,
        f"  Força principal: {exp.get('top_strength','N/A')}",
        f"  Ponto crítico:   {exp.get('top_weakness','N/A')}",
        "",
        "=" * 60,
        "Relatório gerado automaticamente pelo TechNova HR System",
        "=" * 60,
    ]
    return "\n".join(lines)


def render_compare():
    st.markdown("## 📊 Comparação de Candidatos")
    st.divider()

    all_cands = st.session_state.get("all_candidates", [])
    if not all_cands:
        st.info(
            "Nenhum candidato entrevistado ainda. Realize entrevistas para ver o ranking aqui."
        )
        if st.button("🎙️ Iniciar Entrevista"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    # Filtro por vaga
    vagas_avaliadas = list({c["vaga"] for c in all_cands})
    vaga_sel = st.selectbox("Filtrar por vaga:", ["Todas"] + vagas_avaliadas)

    filtrados = (
        all_cands
        if vaga_sel == "Todas"
        else [c for c in all_cands if c["vaga"] == vaga_sel]
    )

    if len(filtrados) < 1:
        st.warning("Nenhum candidato para essa vaga.")
        return

    st.markdown(f"### 🏆 Ranking — {len(filtrados)} candidato(s)")
    df = compare_candidates(filtrados)

    # Exibir tabela com medalhas
    rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
    rec_colors = {"APROVADO": "🟢", "APROVADO_COM_RESSALVAS": "🟡", "REPROVADO": "🔴"}

    for i, row in df.iterrows():
        icon = rank_icons.get(i, f"#{i}")
        rec = row.get("Recomendacao", row.get("Recomendação", "N/A"))
        col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 2, 2, 2, 2])
        col1.markdown(f"### {icon}")
        col2.markdown(f"**{row['Nome']}**  \n{row.get('Perfil Cluster','')}")
        col3.metric("Score Composto", f"{row['Score Composto']:.1f}%")
        col4.metric("Compat. Vaga", f"{row['Compat. Vaga (%)']:.1f}%")
        col5.metric("Fit Cultural", f"{row['Fit Cultural (%)']:.1f}%")
        col6.markdown(f"{rec_colors.get(rec,'⚪')} **{rec.replace('_',' ')}**")
        st.divider()

    # Gráfico de barras comparativo
    st.markdown("### 📊 Visualização Comparativa")
    names = df["Nome"].tolist()
    fig = go.Figure()
    for metric, color in [
        ("Compat. Vaga (%)", GREEN),
        ("Fit Cultural (%)", "#10b981"),
        ("Potencial (%)", "#6ee7b7"),
    ]:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=names,
                y=df[metric].tolist(),
                marker_color=color,
                text=df[metric].apply(lambda v: f"{v:.0f}%"),
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=DARK),
        yaxis=dict(range=[0, 110]),
        title="Comparativo de Scores por Candidato",
    )
    st.plotly_chart(fig, use_container_width=True)

   
    df_export = df.copy()
    df_export = df_export.sort_values(by="Score Composto", ascending=False)
    df_export.insert(0, "Posicao", range(1, len(df_export) + 1))

# Converter para formato correto de porcentagem (0.10 = 10%)
    for col in [
        "Score Composto",
        "Compat. Vaga (%)",
        "Fit Cultural (%)",
        "Potencial (%)",
        "Risco Saida (%)"
    ]:
        if col in df_export.columns:
         df_export[col] = df_export[col] / 100

# =========================
# GERAR EXCEL EM MEMÓRIA
# =========================
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Ranking")

        ws = writer.sheets["Ranking"]

    # Formatar colunas como %
        for col_idx, col_name in enumerate(df_export.columns, 1):
            if col_name in [
                "Score Composto",
                "Compat. Vaga (%)",
                "Fit Cultural (%)",
                "Potencial (%)",
                "Risco Saida (%)"
            ]:
                for row in range(2, len(df_export) + 2):
                    ws.cell(row=row, column=col_idx).number_format = "0.0%"

# =========================
# DOWNLOAD
# =========================
    st.download_button(
        "⬇️ Exportar Ranking (.xlsx)",
        data=output.getvalue(),
        file_name="ranking_candidatos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_dashboard():
    st.markdown("## 📈 Dashboard Analytics")
    st.divider()

    try:
        df = pd.read_csv("data/hr_dataset.csv")
    except FileNotFoundError:
        st.error(
            "Dataset não encontrado. Execute `python src/generate_dataset.py` primeiro."
        )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Funcionários", f"{len(df):,}")
    c2.metric("Attrition", f"{df['Attrition'].mean():.1%}")
    c3.metric("Salário Médio", f"R$ {df['MonthlyIncome'].mean():,.0f}")
    c4.metric("Satisfação", f"{df['JobSatisfaction'].mean():.1f}/4")
    c5.metric("Culture Fit", f"{df['CultureFitScore'].mean():.1f}/10")

    st.divider()
    c1, c2 = st.columns(2)
    dept_filter = c1.multiselect(
        "Departamento",
        df["Department"].unique(),
        default=list(df["Department"].unique()),
    )
    att_filter = c2.selectbox("Attrition", ["Todos", "Ficaram", "Sairam"])

    dff = df[df["Department"].isin(dept_filter)]
    if att_filter == "Ficaram":
        dff = dff[dff["Attrition"] == 0]
    elif att_filter == "Sairam":
        dff = dff[dff["Attrition"] == 1]

    tpl = dict(paper_bgcolor="white", plot_bgcolor="white", font=dict(color=DARK))
    gp = ["#059669", "#10b981", "#6ee7b7", "#047857"]

    c1, c2 = st.columns(2)
    fig = px.histogram(
        dff,
        x="Age",
        color="Department",
        barmode="overlay",
        opacity=0.7,
        title="Distribuição de Idade por Departamento",
        color_discrete_sequence=gp,
    )
    fig.update_layout(**tpl)
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        dff,
        x="Department",
        y="MonthlyIncome",
        color="Department",
        title="Salário por Departamento",
        color_discrete_sequence=gp,
    )
    fig.update_layout(**tpl)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    abd = dff.groupby("Department")["Attrition"].mean().reset_index()
    fig = px.bar(
        abd,
        x="Department",
        y="Attrition",
        title="Taxa de Attrition por Departamento",
        color="Attrition",
        color_continuous_scale=["#d1fae5", "#059669"],
    )
    fig.update_layout(yaxis_tickformat=".0%", **tpl)
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        dff,
        x="CultureFitScore",
        y="EngagementScore",
        color="Attrition",
        size="MonthlyIncome",
        title="Culture Fit vs Engajamento",
        opacity=0.6,
        color_discrete_map={0: GREEN, 1: RED},
    )
    fig.update_layout(**tpl)
    c2.plotly_chart(fig, use_container_width=True)

    # Novo gráfico: distribuição de scores customizados
    st.divider()
    st.markdown("### Distribuição de Scores de Competências")
    score_cols = [
        "SoftSkillsScore",
        "TechnicalSkillsScore",
        "LeadershipScore",
        "CommunicationScore",
        "AdaptabilityScore",
        "CultureFitScore",
    ]
    score_labels = [
        "Soft Skills",
        "Técnico",
        "Liderança",
        "Comunicação",
        "Adaptabilidade",
        "Fit Cultural",
    ]
    fig = go.Figure()
    for col, label in zip(score_cols, score_labels):
        if col in dff.columns:
            fig.add_trace(
                go.Box(y=dff[col], name=label, marker_color=GREEN, boxmean=True)
            )
    fig.update_layout(
        height=400,
        showlegend=False,
        title="Distribuição dos Scores da Base de RH",
        **tpl,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Dados Brutos"):
        st.dataframe(dff, use_container_width=True, height=400)


def main():
    init_session_state()
    render_sidebar()
    page_fn = {
        "home": render_home,
        "culture": render_culture,
        "interview": render_interview,
        "compare": render_compare,
        "dashboard": render_dashboard,
    }
    page_fn.get(st.session_state["page"], render_home)()


if __name__ == "__main__":
    main()
