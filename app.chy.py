import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION ET STYLE
# ============================================
st.set_page_config(
    page_title="Plateforme d'Analyse Financière | Projet Mathématiques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour un look "Bloomberg/TradingView"
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2a2e39; }
    div[data-testid="stExpander"] { background-color: #1e222d; border: none; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# SECTION 1: ACQUISITION DES DONNÉES (6.1)
# ============================================
st.sidebar.header("⚙️ CONFIGURATION DU MARCHÉ")

with st.sidebar:
    actif = st.text_input("Symbole de l'actif (ex: BTC-USD, AAPL, MSFT)", value="BTC-USD").upper()
    
    col1, col2 = st.columns(2)
    date_debut = col1.date_input("Date début", datetime.now() - timedelta(days=365))
    date_fin = col2.date_input("Date fin", datetime.now())
    
    frequence = st.selectbox("Fréquence", ["1d", "1h", "15m", "5m"], index=0)
    
    st.markdown("---")
    st.subheader("Paramètres de la Stratégie")
    sma_court = st.number_input("SMA Courte (Période)", value=20, min_value=5)
    sma_long = st.number_input("SMA Longue (Période)", value=50, min_value=10)
    capital_initial = st.number_input("Capital Initial ($)", value=1000)
    frais_tx = st.slider("Frais de transaction (%)", 0.0, 0.5, 0.1) / 100

@st.cache_data(ttl=3600)
def charger_donnees(ticker, debut, fin, interval):
    try:
        data = yf.download(ticker, start=debut, end=fin, interval=interval, progress=False)
        if data.empty:
            return None
        # Nettoyage des colonnes Multi-index si présentes (yfinance v0.2.40+)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.reset_index()
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

# ============================================
# SECTION 2: TRAITEMENT MATHÉMATIQUE (6.2 & 6.3)
# ============================================

def calculer_metriques_math(df):
    """
    Calcul des rendements et statistiques selon les formules du projet.
    """
    # 1. Rendements (6.2)
    df['R_Arith'] = df['Close'].pct_change()
    df['R_Log'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Statistiques Descriptives (6.3)
    r = df['R_Arith'].dropna()
    
    stats_dict = {
        "Moyenne (Quotidienne)": r.mean(),
        "Médiane": r.median(),
        "Écart-type (σ)": r.std(),
        "Volatilité Annualisée": r.std() * np.sqrt(252),
        "Skewness (Asymétrie)": stats.skew(r),
        "Kurtosis (Aplatissement)": stats.kurtosis(r),
        "Maximum": r.max(),
        "Minimum": r.min(),
        "Percentile 5% (VaR)": r.quantile(0.05),
        "Percentile 25%": r.quantile(0.25),
        "Percentile 75%": r.quantile(0.75),
        "Percentile 95%": r.quantile(0.95)
    }
    return df, stats_dict

# ============================================
# SECTION 3: INDICATEURS TECHNIQUES (6.4)
# ============================================

def ajouter_indicateurs(df):
    # SMA (Moyenne Mobile Simple)
    df['SMA_C'] = df['Close'].rolling(window=sma_court).mean()
    df['SMA_L'] = df['Close'].rolling(window=sma_long).mean()
    
    # Bandes de Bollinger
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (2 * std_20)
    df['BB_Lower'] = df['BB_Mid'] - (2 * std_20)
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ============================================
# SECTION 4: BACKTESTING (6.5)
# ============================================

def executer_backtest(df, cap_init, fees):
    # Signal: 1 si SMA_C > SMA_L, sinon 0 (6.5.1)
    df['Signal'] = np.where(df['SMA_C'] > df['SMA_L'], 1, 0)
    
    # Position décalée pour éviter le look-ahead bias
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements de la stratégie (6.5.2)
    # R_strat = Position_{t-1} * R_actif_t
    df['Strat_Ret_Brut'] = df['Position'] * df['R_Arith']
    
    # Calcul des frais (0.1% par transaction)
    df['Trade'] = df['Position'].diff().abs() # 1 si on change de position
    df['Frais'] = df['Trade'] * fees
    df['Strat_Ret_Net'] = df['Strat_Ret_Brut'] - df['Frais']
    
    # Évolution du capital (6.5.2 pt 4)
    df['Equity'] = cap_init * (1 + df['Strat_Ret_Net'].fillna(0)).cumprod()
    df['BuyHold_Equity'] = cap_init * (1 + df['R_Arith'].fillna(0)).cumprod()
    
    # Métriques de performance (6.5.2 pt 5)
    r_strat = df['Strat_Ret_Net'].dropna()
    total_ret = (df['Equity'].iloc[-1] - cap_init) / cap_init
    vol_ann = r_strat.std() * np.sqrt(252)
    sharpe = (r_strat.mean() / r_strat.std()) * np.sqrt(252) if r_strat.std() != 0 else 0
    
    # Drawdown Maximum
    peak = df['Equity'].cummax()
    dd = (df['Equity'] - peak) / peak
    mdd = dd.min()
    
    performance = {
        "Rendement Total": total_ret,
        "Volatilité Stratégie": vol_ann,
        "Ratio de Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Nombre de Trades": int(df['Trade'].sum())
    }
    
    return df, performance

# ============================================
# INTERFACE PRINCIPALE (DASHBOARD)
# ============================================

st.title(f"📊 Dashboard: {actif}")
st.markdown(f"*Analyse de la période {date_debut} au {date_fin}*")

df_raw = charger_donnees(actif, date_debut, date_fin, frequence)

if df_raw is not None:
    # Traitement des données
    df, stats_math = calculer_metriques_math(df_raw)
    df = ajouter_indicateurs(df)
    df, perf = executer_backtest(df, capital_initial, frais_tx)
    
    # Onglets pour l'organisation (Section 7 du projet)
    tab_main, tab_stats, tab_backtest = st.tabs(["📈 Graphique Principal", "🧮 Analyse Mathématique", "🎯 Backtesting"])
    
    # --- ONGLET 1: GRAPHIQUE PRINCIPAL ---
    with tab_main:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.7, 0.3],
                           subplot_titles=("Prix & Moyennes Mobiles", "Indicateur RSI"))
        
        # Chandeliers ou Ligne de prix
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prix de Clôture", line=dict(color='#2962ff', width=2)), row=1, col=1)
        
        # SMA
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_C'], name=f"SMA {sma_court}", line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_L'], name=f"SMA {sma_long}", line=dict(color='#4caf50', width=1.5)), row=1, col=1)
        
        # Bandes de Bollinger (Optionnel dans la vue)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash'), fill='tonexty', showlegend=False), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='#9c27b0')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", row=2, col=1)
        
        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # --- ONGLET 2: ANALYSE MATHÉMATIQUE ---
    with tab_stats:
        st.header("Analyse Statistique des Rendements")
        
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            st.subheader("Métriques Descriptives")
            # Transformation du dictionnaire en DataFrame pour un affichage propre
            stats_df = pd.DataFrame.from_dict(stats_math, orient='index', columns=['Valeur'])
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
            
            # Test de Normalité (6.3)
            stat, p_val = stats.shapiro(df['R_Arith'].dropna().iloc[:5000]) # Shapiro limité à 5000 points
            st.metric("Test Shapiro-Wilk (p-value)", f"{p_val:.4f}")
            if p_val < 0.05:
                st.error("⚠️ Distribution non normale (p < 0.05)")
            else:
                st.success("✅ Distribution normale")

        with col_s2:
            st.subheader("Distribution des Rendements")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df['R_Arith'].dropna(), nbinsx=50, name="Frequence", marker_color='#2962ff', opacity=0.7))
            fig_hist.update_layout(template="plotly_dark", xaxis_title="Rendement", yaxis_title="Nombre de jours")
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- ONGLET 3: BACKTESTING ---
    with tab_backtest:
        st.header("Résultats de la Stratégie SMA Cross")
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendement Stratégie", f"{perf['Rendement Total']:.2%}")
        m2.metric("Ratio de Sharpe", f"{perf['Ratio de Sharpe']:.2f}")
        m3.metric("Max Drawdown", f"{perf['Max Drawdown']:.2%}")
        m4.metric("Nombre de Trades", perf['Nombre de Trades'])
        
        # Equity Curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df['Date'], y=df['Equity'], name="Stratégie (Net de frais)", line=dict(color='#00e676', width=3)))
        fig_equity.add_trace(go.Scatter(x=df['Date'], y=df['BuyHold_Equity'], name="Buy & Hold (Référence)", line=dict(color='#757575', dash='dot')))
        
        fig_equity.update_layout(title="Évolution du Capital ($)", template="plotly_dark", height=450)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Analyse critique (7.4)
        st.subheader("💡 Analyse Critique")
        diff = perf['Rendement Total'] - ((df['Close'].iloc[-1]/df['Close'].iloc[0]) - 1)
        if diff > 0:
            st.success(f"La stratégie a surperformé le marché de {diff:.2%}. Le ratio de Sharpe de {perf['Ratio de Sharpe']:.2f} suggère une gestion du risque {'efficace' if perf['Ratio de Sharpe'] > 1 else 'modérée'}.")
        else:
            st.warning(f"La stratégie a sous-performé le marché. Les frais de transaction ({frais_tx*100}%) et la latence des signaux SMA peuvent expliquer ce résultat.")

    # Footer
    st.markdown("---")
    st.caption(f"Projet Mathématiques Appliquées | Encadrant: M. Hamza Saber | Données: Yahoo Finance API")

else:
    st.error("Impossible de récupérer les données pour ce symbole. Vérifiez la connexion ou le ticker.")
