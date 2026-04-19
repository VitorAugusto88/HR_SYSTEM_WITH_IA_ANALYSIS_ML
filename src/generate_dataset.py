"""
Gerador de dataset HR baseado no IBM HR Employee Attrition & Performance.
Simula o dataset público do Kaggle com enriquecimento de dados adicionais.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1470  # Mesmo tamanho do dataset IBM original


def generate_dataset():
    """Gera dataset de HR com features originais + enriquecidas."""

    # --- Features originais do IBM HR ---
    age = np.random.randint(18, 60, N)
    gender = np.random.choice(["Male", "Female"], N, p=[0.6, 0.4])
    marital_status = np.random.choice(
        ["Single", "Married", "Divorced"], N, p=[0.32, 0.46, 0.22]
    )
    education = np.random.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.20, 0.35, 0.25, 0.10])
    education_field = np.random.choice(
        ["Life Sciences", "Medical", "Marketing", "Technical Degree",
         "Human Resources", "Other"],
        N, p=[0.37, 0.15, 0.10, 0.20, 0.08, 0.10]
    )

    department = np.random.choice(
        ["Sales", "Research & Development", "Human Resources", "Technology"],
        N, p=[0.25, 0.30, 0.10, 0.35]
    )
    job_role = []
    for dept in department:
        if dept == "Sales":
            job_role.append(np.random.choice(
                ["Sales Executive", "Sales Representative", "Manager"]
            ))
        elif dept == "Research & Development":
            job_role.append(np.random.choice(
                ["Research Scientist", "Laboratory Technician",
                 "Manufacturing Director", "Research Director"]
            ))
        elif dept == "Human Resources":
            job_role.append(np.random.choice(
                ["Human Resources", "HR Analyst", "HR Manager"]
            ))
        else:
            job_role.append(np.random.choice(
                ["Software Engineer", "Data Scientist", "DevOps Engineer",
                 "Tech Lead", "Product Manager"]
            ))

    job_level = np.random.choice([1, 2, 3, 4, 5], N, p=[0.25, 0.30, 0.25, 0.12, 0.08])

    monthly_income = []
    for level in job_level:
        base = {1: 2500, 2: 5000, 3: 8500, 4: 13000, 5: 18000}[level]
        monthly_income.append(int(base + np.random.normal(0, base * 0.2)))
    monthly_income = np.clip(monthly_income, 1500, 25000)

    years_at_company = np.random.exponential(5, N).astype(int).clip(0, 35)
    total_working_years = (years_at_company + np.random.randint(0, 15, N)).clip(0, 40)
    years_in_current_role = np.minimum(
        np.random.exponential(3, N).astype(int), years_at_company
    )
    years_since_last_promotion = np.minimum(
        np.random.exponential(2, N).astype(int), years_at_company
    )
    years_with_curr_manager = np.minimum(
        np.random.exponential(3, N).astype(int), years_at_company
    )
    num_companies_worked = np.random.poisson(2.5, N).clip(0, 9)
    training_times_last_year = np.random.randint(0, 7, N)

    distance_from_home = np.random.exponential(8, N).astype(int).clip(1, 29)
    business_travel = np.random.choice(
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        N, p=[0.15, 0.65, 0.20]
    )
    over_time = np.random.choice(["Yes", "No"], N, p=[0.28, 0.72])

    environment_satisfaction = np.random.choice([1, 2, 3, 4], N, p=[0.10, 0.20, 0.35, 0.35])
    job_satisfaction = np.random.choice([1, 2, 3, 4], N, p=[0.10, 0.20, 0.35, 0.35])
    relationship_satisfaction = np.random.choice([1, 2, 3, 4], N, p=[0.10, 0.20, 0.35, 0.35])
    work_life_balance = np.random.choice([1, 2, 3, 4], N, p=[0.08, 0.22, 0.40, 0.30])
    job_involvement = np.random.choice([1, 2, 3, 4], N, p=[0.05, 0.20, 0.40, 0.35])

    performance_rating = np.random.choice([1, 2, 3, 4], N, p=[0.02, 0.08, 0.55, 0.35])
    percent_salary_hike = np.random.randint(11, 25, N)
    stock_option_level = np.random.choice([0, 1, 2, 3], N, p=[0.40, 0.30, 0.20, 0.10])

    # Attrition baseada em fatores correlacionados
    attrition_prob = np.zeros(N)
    attrition_prob += (over_time == "Yes") * 0.15
    attrition_prob += (job_satisfaction <= 2) * 0.12
    attrition_prob += (environment_satisfaction <= 2) * 0.10
    attrition_prob += (work_life_balance <= 2) * 0.10
    attrition_prob += (years_at_company < 2) * 0.08
    attrition_prob += (monthly_income < np.array(monthly_income).mean()) * 0.05
    attrition_prob += (distance_from_home > 15) * 0.05
    attrition_prob += (num_companies_worked > 4) * 0.08
    attrition_prob += (age < 30) * 0.05
    attrition_prob += (business_travel == "Travel_Frequently") * 0.06
    attrition_prob += (years_since_last_promotion > 5) * 0.07
    attrition_prob = np.clip(attrition_prob + np.random.normal(0, 0.05, N), 0.02, 0.95)
    attrition = (np.random.random(N) < attrition_prob).astype(int)

    # --- Features ENRIQUECIDAS ---
    soft_skills_score = np.random.normal(7, 1.5, N).clip(1, 10).round(1)
    technical_skills_score = np.random.normal(7, 1.5, N).clip(1, 10).round(1)
    leadership_score = np.random.normal(6, 2, N).clip(1, 10).round(1)
    communication_score = np.random.normal(7, 1.5, N).clip(1, 10).round(1)
    adaptability_score = np.random.normal(6.5, 1.8, N).clip(1, 10).round(1)
    innovation_score = np.random.normal(6, 2, N).clip(1, 10).round(1)
    teamwork_score = np.random.normal(7.2, 1.3, N).clip(1, 10).round(1)

    # Culture fit score (baseado em valores da empresa)
    culture_fit_score = (
        0.20 * adaptability_score +
        0.20 * teamwork_score +
        0.15 * communication_score +
        0.15 * innovation_score +
        0.10 * soft_skills_score +
        0.10 * (work_life_balance / 4 * 10) +
        0.10 * (relationship_satisfaction / 4 * 10)
    ).round(1)

    # Engagement score
    engagement_score = (
        0.25 * (job_satisfaction / 4 * 10) +
        0.25 * (job_involvement / 4 * 10) +
        0.20 * (environment_satisfaction / 4 * 10) +
        0.15 * (work_life_balance / 4 * 10) +
        0.15 * (relationship_satisfaction / 4 * 10)
    ).round(1)

    # Potencial de crescimento
    growth_potential = (
        0.25 * technical_skills_score +
        0.20 * leadership_score +
        0.20 * adaptability_score +
        0.15 * innovation_score +
        0.10 * (training_times_last_year / 6 * 10) +
        0.10 * (performance_rating / 4 * 10)
    ).round(1)

    # Salary competitiveness (comparação com mercado)
    market_salary = {1: 3000, 2: 6000, 3: 10000, 4: 15000, 5: 20000}
    salary_competitiveness = np.array([
        (inc / market_salary[lvl]) * 100
        for inc, lvl in zip(monthly_income, job_level)
    ]).round(1)

    df = pd.DataFrame({
        "EmployeeID": range(1, N + 1),
        "Age": age,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "Education": education,
        "EducationField": education_field,
        "Department": department,
        "JobRole": job_role,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "TotalWorkingYears": total_working_years,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "NumCompaniesWorked": num_companies_worked,
        "TrainingTimesLastYear": training_times_last_year,
        "DistanceFromHome": distance_from_home,
        "BusinessTravel": business_travel,
        "OverTime": over_time,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobSatisfaction": job_satisfaction,
        "RelationshipSatisfaction": relationship_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "JobInvolvement": job_involvement,
        "PerformanceRating": performance_rating,
        "PercentSalaryHike": percent_salary_hike,
        "StockOptionLevel": stock_option_level,
        "Attrition": attrition,
        # Features enriquecidas
        "SoftSkillsScore": soft_skills_score,
        "TechnicalSkillsScore": technical_skills_score,
        "LeadershipScore": leadership_score,
        "CommunicationScore": communication_score,
        "AdaptabilityScore": adaptability_score,
        "InnovationScore": innovation_score,
        "TeamworkScore": teamwork_score,
        "CultureFitScore": culture_fit_score,
        "EngagementScore": engagement_score,
        "GrowthPotential": growth_potential,
        "SalaryCompetitiveness": salary_competitiveness,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/hr_dataset.csv", index=False)
    print(f"Dataset gerado: {df.shape[0]} registros, {df.shape[1]} colunas")
    print(f"Taxa de Attrition: {df['Attrition'].mean():.1%}")
    print(f"\nColunas: {list(df.columns)}")
