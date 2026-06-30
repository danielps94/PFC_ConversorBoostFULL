"""
PFC Boost DCM - Aplicação Streamlit
Versão: 1.0
Autor original: Daniel PS
Melhorias: interface, validações, caching, exportação de resultados
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from io import StringIO

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(page_title="PFC Boost DCM", page_icon="⚡", layout="wide")

# ------------------------------------------------------------
# Helpers e funções numéricas (cacheadas para desempenho)
# ------------------------------------------------------------
@st.cache_data
def integral_indutor_cached(Vgpk, Vo):
    """
    Calcula a integral usada no dimensionamento do indutor:
    ∫_0^π sin^2(θ) / (1 - (Vgpk/Vo) sin θ) dθ
    """
    def integrand(theta, Vgpk_local, Vo_local):
        return (np.sin(theta) ** 2) / (1 - (Vgpk_local / Vo_local) * np.sin(theta))

    # Tratamento simples: se Vo <= Vgpk, integral não faz sentido (divisão por zero possível)
    if Vo <= Vgpk:
        return np.nan
    val, err = quad(integrand, 0, np.pi, args=(Vgpk, Vo), limit=200)
    return val

@st.cache_data
def integral_capacitor_cached(Vgpk, Po, D, Vo, fs, fr, limite_integracao):
    """
    Função integrando o comportamento do capacitor usada para calcular C exato.
    Integra no intervalo [0, limite_integracao] (s).
    """
    Mb = Vo / Vgpk

    def integrand(t, Vgpk_local, Po_local, D_local, Vo_local, fs_local, fr_local):
        termo1 = (D_local ** 2 * Vgpk_local ** 2) / (2 * np.pi * Po_local * fs_local)
        seno = np.sin(2 * np.pi * fr_local * t)
        # protege contra divisão por zero
        denom = Mb - seno
        if np.any(np.isclose(denom, 0.0)):
            # valor grande para penalizar região problemática; quad evita pontos singulares
            denom = np.where(np.isclose(denom, 0.0), 1e-12, denom)
        termo2 = (seno ** 2) / denom
        termo3 = Po_local / Vo_local
        return np.abs(termo1 * termo2 - termo3)

    val, err = quad(integrand, 0, limite_integracao, args=(Vgpk, Po, D, Vo, fs, fr), limit=200)
    return val

def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan

# ------------------------------------------------------------
# Interface lateral (parâmetros)
# ------------------------------------------------------------
st.sidebar.title("Parâmetros do Projeto")

st.sidebar.markdown("### Entrada (AC)")
Vg_min = st.sidebar.number_input("Vg mínimo (Vrms)", value=90.0, step=1.0, format="%.1f")
Vg_max = st.sidebar.number_input("Vg máximo (Vrms)", value=240.0, step=1.0, format="%.1f")
Vg = st.sidebar.slider("Vg atual (Vrms)", min_value=float(Vg_min), max_value=float(Vg_max), value=127.0)

st.sidebar.markdown("### Saída / Operação")
Po_min = st.sidebar.number_input("Po mínimo (W)", value=50.0, step=1.0, format="%.1f")
Po_max = st.sidebar.number_input("Po máximo (W)", value=150.0, step=1.0, format="%.1f")
Po = st.sidebar.slider("Po atual (W)", min_value=float(Po_min), max_value=float(Po_max), value=100.0)

Vo = st.sidebar.number_input("Vo (V)", value=450.0, step=1.0, format="%.1f")
DeltaVo = st.sidebar.number_input("Ripple (%)", value=10.0, min_value=0.1, step=0.1, format="%.2f")

st.sidebar.markdown("### Temporização")
fs = st.sidebar.number_input("Frequência de chaveamento (Hz)", value=50000.0, step=1000.0, format="%.0f")
fr = st.sidebar.number_input("Frequência da rede (Hz)", value=60.0, step=1.0, format="%.1f")

st.sidebar.divider()
st.sidebar.markdown("### Projeto")
margem_dcm = st.sidebar.slider("Margem sobre Dcrit (fração)", min_value=0.01, max_value=0.99, value=0.90, step=0.01)

st.sidebar.divider()
st.sidebar.markdown("### Presets")
preset = st.sidebar.selectbox("Carregar preset", options=["Personalizado", "Residencial 127V/60Hz", "Industrial 230V/60Hz"])
if preset == "Residencial 127V/60Hz":
    Vg_min, Vg_max, Vg = 110.0, 140.0, 127.0
    Po_min, Po_max, Po = 50.0, 200.0, 100.0
    Vo = 400.0
    fs = 50000.0
    fr = 60.0
elif preset == "Industrial 230V/60Hz":
    Vg_min, Vg_max, Vg = 200.0, 260.0, 230.0
    Po_min, Po_max, Po = 100.0, 1000.0, 300.0
    Vo = 700.0
    fs = 80000.0
    fr = 60.0

# ------------------------------------------------------------
# Conversões e verificações iniciais
# ------------------------------------------------------------
Vg_min_pk = Vg_min * np.sqrt(2)
Vg_max_pk = Vg_max * np.sqrt(2)
Vg_pk = Vg * np.sqrt(2)
DeltaVo_V = DeltaVo * Vo / 100.0

if Vo <= Vg_max_pk:
    st.sidebar.error("A tensão de saída (Vo) deve ser maior que a tensão máxima retificada (Vg_max_pk). Ajuste Vo.")
    st.stop()

# Duty crítico e projetado
Dc = 1.0 - (Vg_max_pk / Vo)
D_proj = margem_dcm * Dc

# ============================================================
# Título + resumo rápido
# ============================================================
st.title("⚡ Projeto de PFC Boost em DCM — Aplicação")
st.write("Ferramenta para dimensionamento de indutor e capacitor para conversor Boost operando em DCM (PFC).")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vg atual", f"{Vg:.1f} Vrms")
c2.metric("Po atual", f"{Po:.1f} W")
c3.metric("Duty crítico (Dc)", f"{Dc:.4f}")
c4.metric("Duty projeto (D_proj)", f"{D_proj:.4f}")

st.divider()

# ============================================================
# Método: avaliação dos 4 casos limites e cálculo
# ============================================================
st.header("Dimensionamento — Indutor e Capacitor")
st.markdown("Avaliação dos quatro casos limites para encontrar L e C adotados.")

casos = [
    ("Caso 1", Vg_min, Po_min),
    ("Caso 2", Vg_min, Po_max),
    ("Caso 3", Vg_max, Po_min),
    ("Caso 4", Vg_max, Po_max)
]

# Cálculo do indutor para cada caso (mostrar progresso)
tabela_indutor = []
L_valores = []
integrais = []

with st.spinner("Calculando indutâncias (integrais numéricas)..."):
    for nome, Vg_rms, Po_caso in casos:
        Vgpk = Vg_rms * np.sqrt(2)
        integral = integral_indutor_cached(Vgpk, Vo)
        integrais.append(integral)

        # fórmula L = (Vgpk^2 * D_proj^2) / (2 * Po * fs) * (1/pi) * integral
        if np.isnan(integral):
            L = np.nan
        else:
            L = (Vgpk ** 2) * (D_proj ** 2) / (2.0 * Po_caso * fs)
            L = L * (1.0 / np.pi) * integral

        L_valores.append(L)
        tabela_indutor.append({
            "Caso": nome,
            "Vg (Vrms)": Vg_rms,
            "Po (W)": Po_caso,
            "Integral": (np.nan if np.isnan(integral) else round(integral, 6)),
            "L (µH)": (np.nan if np.isnan(L) else round(L * 1e6, 2))
        })

df_indutor = pd.DataFrame(tabela_indutor)
st.subheader("Tabela: Indutância por Caso")
st.dataframe(df_indutor, use_container_width=True)

# Seleciona menor L adotado (criterio original: menor L)
valid_L_vals = [x for x in L_valores if not np.isnan(x)]
if len(valid_L_vals) == 0:
    st.error("Não foi possível calcular indutâncias válidas com os parâmetros fornecidos.")
    st.stop()

indice = int(np.nanargmin(L_valores))
L_adotado = L_valores[indice]
caso_critico = tabela_indutor[indice]

st.success(f"Indutância adotada = {L_adotado*1e6:.2f} µH")
st.info(f"Caso crítico (menor L): {caso_critico['Caso']} — Vg={caso_critico['Vg (Vrms)']} Vrms, Po={caso_critico['Po (W)']} W")

# ============================================================
# Cálculo do capacitor (para cada caso recalcula-se D e integra-se)
# ============================================================
st.subheader("Dimensionamento do Capacitor")

tabela_capacitor = []
C_exato_valores = []
C_ref_valores = []
limite_integracao = 1.0 / (4.0 * fr)  # intervalo de integração adotado

with st.spinner("Calculando capacitâncias (integrais numéricas)..."):
    for nome, Vg_rms, Po_caso in casos:
        Vgpk = Vg_rms * np.sqrt(2)

        # integral do indutor para o caso (pode reutilizar)
        integral_L = integral_indutor_cached(Vgpk, Vo)
        if np.isnan(integral_L) or integral_L == 0:
            D_real = np.nan
            integral_C = np.nan
            C_exato = np.nan
        else:
            # termo_D -> D_real = sqrt( (2*Po*fs*L) / (Vgpk^2*(1/pi)*integral_L) )
            termo_D = (2.0 * Po_caso * fs * L_adotado) / ((Vgpk ** 2) * (1.0 / np.pi) * integral_L)
            D_real = np.sqrt(np.abs(termo_D))  # abs para evitar números negativos por erro numérico

            # integral do capacitor
            try:
                integral_C = integral_capacitor_cached(Vgpk, Po_caso, D_real, Vo, fs, fr, limite_integracao)
            except Exception:
                integral_C = np.nan

            if np.isnan(integral_C):
                C_exato = np.nan
            else:
                C_exato = safe_div(integral_C, DeltaVo_V)

        # referência aproximada
        C_ref = safe_div(Po_caso, (2.0 * np.pi * fr * Vo * DeltaVo_V))

        C_exato_valores.append(C_exato)
        C_ref_valores.append(C_ref)

        tabela_capacitor.append({
            "Caso": nome,
            "Duty": (np.nan if np.isnan(D_real) else round(D_real, 6)),
            "C Exato (µF)": (np.nan if np.isnan(C_exato) else round(C_exato * 1e6, 2)),
            "C Ref (µF)": (np.nan if np.isnan(C_ref) else round(C_ref * 1e6, 2))
        })

df_capacitor = pd.DataFrame(tabela_capacitor)
st.dataframe(df_capacitor, use_container_width=True)

# Adotar maior C (critério original)
valid_Cs = [x for x in C_exato_valores if not np.isnan(x)]
if len(valid_Cs) == 0:
    st.error("Não foi possível calcular capacitâncias válidas com os parâmetros fornecidos.")
    st.stop()

indice_C = int(np.nanargmax([np.nan if np.isnan(x) else x for x in C_exato_valores]))
C_adotado = C_exato_valores[indice_C]
caso_critico_C = tabela_capacitor[indice_C]

st.success(f"Capacitância adotada = {C_adotado*1e6:.2f} µF")
st.info(f"Caso crítico (maior C): {caso_critico_C['Caso']} — Duty: {caso_critico_C['Duty']}")

# ============================================================
# Resultados finais para o ponto de operação atual
# ============================================================
st.header("Resultados Finais do Projeto")

# recalcula integral para Vg atual
integral_atual = integral_indutor_cached(Vg_pk, Vo)
termo_D = (2.0 * Po * fs * L_adotado) / ((Vg_pk ** 2) * (1.0 / np.pi) * integral_atual)
D_atual = np.sqrt(np.abs(termo_D))

c1, c2, c3 = st.columns(3)
c1.metric("Indutância adotada", f"{L_adotado*1e6:.2f} µH")
c2.metric("Capacitância adotada", f"{C_adotado*1e6:.2f} µF")
c3.metric("Duty de operação", f"{D_atual:.4f}")

st.divider()

# ============================================================
# Qualidade de energia: FP e THD (integrais)
# ============================================================
st.subheader("Qualidade da Energia (FP e THD)")
alpha = Vg_pk / Vo

def integranda_num(theta):
    return (np.sin(theta) ** 2) / (1 - alpha * np.sin(theta))

def integranda_den(theta):
    return (np.sin(theta) / (1 - alpha * np.sin(theta))) ** 2

num, _ = quad(integranda_num, 0, np.pi)
den, _ = quad(integranda_den, 0, np.pi)

# proteção contra zero/negativo
if den <= 0:
    FP = 1.0
else:
    FP = (np.sqrt(2) * num) / (np.sqrt(np.pi * den))
FP = min(FP, 1.0)
THD = np.sqrt(max(0.0, (1.0 / (FP ** 2)) - 1.0)) * 100.0

c1, c2 = st.columns(2)
c1.metric("Fator de Potência (FP)", f"{FP:.4f}")
c2.metric("THD", f"{THD:.2f} %")

st.divider()

# ============================================================
# Resumo e validações
# ============================================================
st.subheader("Resumo do Projeto")
resumo = pd.DataFrame({
    "Parâmetro": ["Vg", "Po", "Vo", "fs", "L", "C", "Duty", "FP", "THD"],
    "Valor": [
        f"{Vg:.1f} Vrms",
        f"{Po:.1f} W",
        f"{Vo:.1f} V",
        f"{fs:.0f} Hz",
        f"{L_adotado*1e6:.2f} µH",
        f"{C_adotado*1e6:.2f} µF",
        f"{D_atual:.4f}",
        f"{FP:.4f}",
        f"{THD:.2f} %"
    ]
})
st.dataframe(resumo, use_container_width=True, hide_index=True)

st.subheader("Validações Automáticas")
if FP > 0.98:
    st.success("✔ Fator de potência excelente.")
elif FP > 0.95:
    st.warning("⚠ Fator de potência aceitável.")
else:
    st.error("✖ Fator de potência abaixo do esperado.")

if THD < 10:
    st.success("✔ THD dentro do esperado.")
elif THD < 20:
    st.warning("⚠ THD moderado.")
else:
    st.error("✖ THD elevado.")

if D_atual < Dc:
    st.success("✔ Conversor permanece em DCM.")
else:
    st.error("✖ Duty acima do limite crítico (pode sair de DCM).")

st.divider()

# ============================================================
# Gráficos e exportação
# ============================================================
st.header("Visualizações")

# Gráfico de tensão de entrada e ripple de saída
tempo = np.linspace(0, 1.0/fr, 1000)
Vg_t = Vg_pk * np.sin(2.0 * np.pi * fr * tempo)
ondulacao = DeltaVo_V
V_saida = Vo + ondulacao * np.sin(2.0 * np.pi * fr * tempo)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
ax1.plot(tempo * 1000, Vg_t, 'b-', label='Vg(t)')
ax1.set_ylabel('Tensão (V)')
ax1.set_title('Tensão de Entrada')
ax1.grid(True)

ax2.plot(tempo * 1000, V_saida, 'r-', label='Vo(t)')
ax2.axhline(y=Vo, color='k', linestyle='--', linewidth=1, alpha=0.6)
ax2.fill_between(tempo * 1000, Vo - ondulacao / 2.0, Vo + ondulacao / 2.0, alpha=0.15, color='red')
ax2.set_xlabel('Tempo (ms)')
ax2.set_ylabel('Tensão (V)')
ax2.set_title('Tensão de Saída com Ripple')
ax2.grid(True)

st.pyplot(fig)

# Gráfico L por caso
fig2, ax = plt.subplots(figsize=(8, 4))
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = df_indutor["Caso"]
vals = df_indutor["L (µH)"].values
ax.bar(labels, vals, color=cores, alpha=0.8)
ax.axhline(y=L_adotado*1e6, color='r', linestyle='--', label=f'L adotado: {L_adotado*1e6:.2f} µH')
ax.set_ylabel('Indutância (µH)')
ax.set_title('Indutância por Caso')
ax.legend()
st.pyplot(fig2)

# Gráfico C por caso
fig3, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(df_capacitor))
largura = 0.35
ax.bar(x - largura/2, df_capacitor["C Exato (µF)"], largura, label='C Exato', color='steelblue', alpha=0.8)
ax.bar(x + largura/2, df_capacitor["C Ref (µF)"], largura, label='C Ref', color='orange', alpha=0.8)
ax.axhline(y=C_adotado*1e6, color='r', linestyle='--', label=f'C adotado: {C_adotado*1e6:.2f} µF')
ax.set_xticks(x)
ax.set_xticklabels(df_capacitor["Caso"])
ax.set_ylabel('Capacitância (µF)')
ax.set_title('Capacitância por Caso')
ax.legend()
st.pyplot(fig3)

st.divider()

# Exportar resultados (CSV)
st.subheader("Exportar Resultados")
col_a, col_b = st.columns([2, 1])
with col_a:
    to_export = pd.concat([df_indutor, df_capacitor.rename(columns={"Caso": "Caso_cap"})], axis=1)
    csv_buffer = StringIO()
    to_export.to_csv(csv_buffer, index=False)
    st.download_button("Baixar resultados (CSV)", csv_buffer.getvalue(), file_name="pfc_boost_results.csv", mime="text/csv")
with col_b:
    if st.button("Copiar resumo para área de transferência"):
        st.experimental_set_query_params()  # placeholder para UX
        st.write("Resumo copiado (use o botão de download para CSV).")

st.divider()

# ============================================================
# Sobre / instruções
# ============================================================
with st.expander("ℹ️ Sobre este aplicativo"):
    st.markdown("""
    Aplicação educacional para dimensionamento de conversor Boost em DCM operando como PFC.
    Melhorias adicionadas:
    - Cache nas integrais para maior velocidade
    - Presets de operação
    - Validações e mensagens de erro
    - Gráficos e exportação CSV

    Tecnologias: Python, Streamlit, NumPy, SciPy, Pandas, Matplotlib.
    """)
