"""
Modulo de predicao ML com scores de compatibilidade ponderados por vaga,
analise de fatores explicativos e comparacao entre candidatos.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import POSITION_WEIGHTS

MODELS_DIR = Path(__file__).parent.parent / "models"


class HRPredictor:
    """Encapsula todos os modelos de ML para predicao de RH."""

    def __init__(self):
        self.attrition_model = joblib.load(MODELS_DIR / "attrition_model.joblib")
        self.attrition_threshold = self._load_optional_artifact(
            "attrition_threshold.joblib", 0.5
        )
        self.kmeans_model = joblib.load(MODELS_DIR / "kmeans_model.joblib")
        self.scaler_cluster = joblib.load(MODELS_DIR / "scaler_cluster.joblib")
        self.le_dict = joblib.load(MODELS_DIR / "label_encoders.joblib")
        self.feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
        self.cluster_features = joblib.load(MODELS_DIR / "cluster_features.joblib")
        self.pca = joblib.load(MODELS_DIR / "pca_model.joblib")
        self.cluster_names = self._load_optional_artifact("cluster_names.joblib", {})

    def _load_optional_artifact(self, filename: str, default):
        path = MODELS_DIR / filename
        return joblib.load(path) if path.exists() else default

    def _resolve_cluster_labels(self) -> dict[int, str]:
        default_cluster_names = {
            0: "Profissional Estavel",
            1: "Alto Potencial",
            2: "Em Desenvolvimento",
            3: "Especialista Tecnico",
        }
        raw_cluster_names = self.cluster_names or default_cluster_names

        labels = {}
        counts = {}
        for cluster_id in sorted(raw_cluster_names):
            base_label = raw_cluster_names[cluster_id]
            counts[base_label] = counts.get(base_label, 0) + 1
            if counts[base_label] > 1:
                labels[cluster_id] = f"{base_label} (C{cluster_id})"
            else:
                labels[cluster_id] = base_label
        return labels

    def build_candidate_features(
        self,
        nlp_scores: dict,
        position_info: dict,
        candidate_info: dict = None,
    ) -> pd.DataFrame:
        """Constroi o vetor de features do candidato."""
        scores = nlp_scores["scores"]
        cand = candidate_info or {}

        dept_map = {value: idx for idx, value in enumerate(self.le_dict["Department"].classes_)}
        dept_encoded = dept_map.get(position_info.get("departamento", "Technology"), 0)

        nivel_map = {"Junior": 1, "Pleno": 2, "Sênior": 3, "Lead": 4, "Diretor": 5}

        candidate = {
            "Age": cand.get("idade", 30),
            "Gender": 0,
            "MaritalStatus": 1,
            "Education": cand.get("escolaridade", 3),
            "EducationField": 0,
            "Department": dept_encoded,
            "JobRole": 0,
            "JobLevel": nivel_map.get(position_info.get("nivel", "Pleno"), 2),
            "MonthlyIncome": 8000,
            "YearsAtCompany": 0,
            "TotalWorkingYears": cand.get(
                "experiencia", position_info.get("experiencia_minima_anos", 3)
            ),
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
            "NumCompaniesWorked": cand.get("empresas", 2),
            "TrainingTimesLastYear": 3,
            "DistanceFromHome": 10,
            "BusinessTravel": 1,
            "OverTime": 0,
            "EnvironmentSatisfaction": int(scores.get("job_satisfaction_potential", 3)),
            "JobSatisfaction": int(scores.get("job_satisfaction_potential", 3)),
            "RelationshipSatisfaction": 3,
            "WorkLifeBalance": int(scores.get("work_life_balance_expectation", 3)),
            "JobInvolvement": min(
                4, max(1, round(scores.get("engagement_potential", 7) / 2.5))
            ),
            "PerformanceRating": min(
                4, max(1, round(scores.get("technical_skills", 7) / 2.5))
            ),
            "PercentSalaryHike": 15,
            "StockOptionLevel": 1,
            "SoftSkillsScore": scores.get("soft_skills", 7),
            "TechnicalSkillsScore": scores.get("technical_skills", 7),
            "LeadershipScore": scores.get("leadership", 6),
            "CommunicationScore": scores.get("communication", 7),
            "AdaptabilityScore": scores.get("adaptability", 6.5),
            "InnovationScore": scores.get("innovation", 6),
            "TeamworkScore": scores.get("teamwork", 7),
            "CultureFitScore": scores.get("culture_fit", 7),
            "EngagementScore": scores.get("engagement_potential", 7),
            "GrowthPotential": scores.get("growth_potential", 7),
            "SalaryCompetitiveness": 90.0,
        }
        candidate_df = pd.DataFrame([candidate])
        return candidate_df.reindex(columns=self.feature_cols, fill_value=0)

    def predict_attrition(self, candidate_df: pd.DataFrame) -> dict:
        X = candidate_df[self.feature_cols]
        proba = self.attrition_model.predict_proba(X)[0]
        threshold = float(self.attrition_threshold)
        predicted_attrition = int(proba[1] >= threshold)

        if proba[1] >= max(0.6, threshold):
            risk_level = "ALTO"
        elif proba[1] >= max(0.3, threshold * 0.75):
            risk_level = "MEDIO"
        else:
            risk_level = "BAIXO"

        return {
            "probabilidade_ficar": float(proba[0]),
            "probabilidade_sair": float(proba[1]),
            "threshold_modelo": threshold,
            "predicao_binaria": predicted_attrition,
            "risco": risk_level,
        }

    def predict_cluster(self, candidate_df: pd.DataFrame) -> dict:
        X = candidate_df[self.cluster_features]
        X_scaled = self.scaler_cluster.transform(X)
        cluster = self.kmeans_model.predict(X_scaled)[0]
        cluster_labels = self._resolve_cluster_labels()

        distances = self.kmeans_model.transform(X_scaled)[0]
        fit_score = 1 - (distances[cluster] / (distances.max() + 1e-9))

        return {
            "cluster": int(cluster),
            "perfil": cluster_labels.get(cluster, f"Cluster {cluster}"),
            "fit_score": float(fit_score),
            "distancias": {
                cluster_labels.get(i, f"C{i}"): float(d) for i, d in enumerate(distances)
            },
        }

    def calculate_compatibility_scores(
        self, nlp_scores: dict, position_key: str = None
    ) -> dict:
        """
        Calcula scores de compatibilidade com pesos especificos por vaga.
        Se position_key for fornecido, usa pesos personalizados daquela posicao.
        """
        scores = nlp_scores["scores"]
        weights = POSITION_WEIGHTS.get(
            position_key,
            {
                "technical_skills": 0.35,
                "communication": 0.15,
                "teamwork": 0.15,
                "soft_skills": 0.15,
                "growth_potential": 0.20,
            },
        )

        job_compat = sum(scores.get(k, 5) * w for k, w in weights.items()) / 10

        culture_compat = (
            scores.get("culture_fit", 5) * 0.30
            + scores.get("adaptability", 5) * 0.20
            + scores.get("teamwork", 5) * 0.20
            + scores.get("innovation", 5) * 0.15
            + scores.get("communication", 5) * 0.15
        ) / 10

        growth = (
            scores.get("growth_potential", 5) * 0.30
            + scores.get("leadership", 5) * 0.20
            + scores.get("adaptability", 5) * 0.20
            + scores.get("innovation", 5) * 0.15
            + scores.get("technical_skills", 5) * 0.15
        ) / 10

        return {
            "compatibilidade_vaga": round(job_compat * 100, 1),
            "compatibilidade_cultura": round(culture_compat * 100, 1),
            "potencial_crescimento": round(growth * 100, 1),
        }

    def get_score_explanation(
        self, nlp_scores: dict, position_key: str = None
    ) -> dict:
        """
        Gera uma explicacao dos fatores que mais impactaram cada score.
        Substitui SHAP de forma interpretavel mesmo sem os modelos de explicacao.
        """
        scores = nlp_scores["scores"]
        weights = POSITION_WEIGHTS.get(
            position_key,
            {
                "technical_skills": 0.35,
                "communication": 0.15,
                "teamwork": 0.15,
                "soft_skills": 0.15,
                "growth_potential": 0.20,
            },
        )

        label_map = {
            "technical_skills": "Habilidades Tecnicas",
            "communication": "Comunicacao",
            "teamwork": "Trabalho em Equipe",
            "soft_skills": "Soft Skills",
            "growth_potential": "Potencial de Crescimento",
            "culture_fit": "Fit Cultural",
            "adaptability": "Adaptabilidade",
            "leadership": "Lideranca",
            "innovation": "Inovacao",
            "engagement_potential": "Engajamento",
        }

        contributions = [
            {
                "fator": label_map.get(k, k),
                "score": round(scores.get(k, 5), 1),
                "peso": round(w, 2),
                "contribuicao": round(scores.get(k, 5) * w, 2),
            }
            for k, w in weights.items()
        ]
        contributions.sort(key=lambda x: x["contribuicao"], reverse=True)

        strengths = [c for c in contributions if c["score"] >= 7.5]
        weaknesses = [c for c in contributions if c["score"] < 5.5]

        return {
            "contributions": contributions,
            "top_strength": strengths[0]["fator"] if strengths else "N/A",
            "top_weakness": weaknesses[0]["fator"] if weaknesses else "Nenhuma critica",
        }

    def full_prediction(
        self,
        nlp_scores: dict,
        position_info: dict,
        candidate_info: dict = None,
        position_key: str = None,
    ) -> dict:
        """Executa todas as predicoes para um candidato."""
        candidate_df = self.build_candidate_features(
            nlp_scores, position_info, candidate_info
        )
        attrition = self.predict_attrition(candidate_df)
        cluster = self.predict_cluster(candidate_df)
        compatibility = self.calculate_compatibility_scores(nlp_scores, position_key)
        explanation = self.get_score_explanation(nlp_scores, position_key)

        return {
            "attrition": attrition,
            "cluster": cluster,
            "compatibility": compatibility,
            "explanation": explanation,
            "nlp_evaluation": {
                "resumo": nlp_scores.get("resumo_avaliacao", ""),
                "pontos_fortes": nlp_scores.get("pontos_fortes", []),
                "pontos_atencao": nlp_scores.get("pontos_atencao", []),
                "recomendacao": nlp_scores.get("recomendacao", ""),
                "justificativa": nlp_scores.get("justificativa_recomendacao", ""),
            },
        }


def compare_candidates(candidates: list[dict]) -> pd.DataFrame:
    """
    Recebe lista de dicts com {nome, nlp_analysis, predictions}
    e retorna um DataFrame ordenado por score composto para ranking.
    """
    rows = []
    for candidate in candidates:
        pred = candidate["predictions"]
        nlp = candidate["nlp_analysis"]
        compat = pred["compatibility"]
        rows.append(
            {
                "Nome": candidate["nome"],
                "Recomendacao": nlp.get("recomendacao", "N/A"),
                "Compat. Vaga (%)": compat["compatibilidade_vaga"],
                "Fit Cultural (%)": compat["compatibilidade_cultura"],
                "Risco Saida (%)": round(pred["attrition"]["probabilidade_sair"] * 100, 1),
                "Potencial (%)": compat["potencial_crescimento"],
                "Perfil Cluster": pred["cluster"]["perfil"],
                "Score Composto": round(
                    compat["compatibilidade_vaga"] * 0.35
                    + compat["compatibilidade_cultura"] * 0.25
                    + compat["potencial_crescimento"] * 0.25
                    + (100 - pred["attrition"]["probabilidade_sair"] * 100) * 0.15,
                    1,
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("Score Composto", ascending=False).reset_index(
        drop=True
    )
    df.index += 1
    return df
