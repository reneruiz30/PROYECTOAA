import ast
import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from fpdf import FPDF
from mlxtend.frequent_patterns import apriori, association_rules
from streamlit_supabase_auth import login_form, logout_button
from supabase import create_client

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(page_title="OPSO - Optimal Placement Stock", page_icon="🛒", layout="wide")


# ── CSS externo ─────────────────────────────────────────────────────────────
def load_css(path: Path) -> None:
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(Path(__file__).parent / "style.css")


# ── Session state ───────────────────────────────────────────────────────────
for key in ("user", "df_bruto", "df_cesta", "reglas"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Supabase ────────────────────────────────────────────────────────────────
_url = st.secrets["supabase"]["url"]
_key = st.secrets["supabase"]["key"]
supabase = create_client(_url, _key)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def dibujar_plano(titulo: str, estantes: list, indices_ruta: list | None = None):
    """Genera un plano matplotlib del supermercado con ruta opcional."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax.add_patch(patches.Rectangle((0.5, 0.5), 2.5, 1.5, facecolor="#cccccc", edgecolor="black", linewidth=1.5))
    ax.text(1.75, 1.25, "Cajas", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(patches.Rectangle((7.0, 0.5), 2.5, 1.5, facecolor="#c8e6c9", edgecolor="black", linewidth=1.5))
    ax.text(8.25, 1.25, "Entrada", ha="center", va="center", fontsize=10, fontweight="bold")

    centros_x, centros_y = [], []
    for i, (x, y, ancho, alto, label, color) in enumerate(estantes):
        en_ruta = indices_ruta and i in indices_ruta
        ax.add_patch(patches.Rectangle(
            (x, y), ancho, alto,
            facecolor=color,
            edgecolor="red" if en_ruta else "black",
            linewidth=3 if en_ruta else 1.2,
            alpha=0.9,
        ))
        ax.text(x + ancho / 2, y + alto / 2, label,
                ha="center", va="center", fontsize=8, fontweight="bold", color="black", wrap=True)
        centros_x.append(x + ancho / 2)
        centros_y.append(y + alto / 2)

    if indices_ruta:
        rx = [8.25] + [centros_x[i] for i in indices_ruta] + [1.75]
        ry = [1.25] + [centros_y[i] for i in indices_ruta] + [1.25]
        ax.plot(rx, ry, color="#FF4B4B", linestyle="--", linewidth=2.5, marker="o")

    fig.tight_layout()
    return fig


def calcular_kpis(reglas: pd.DataFrame) -> dict:
    lift_prom      = reglas["lift"].mean()
    conf_prom      = reglas["confidence"].mean()
    fuertes        = len(reglas[reglas["lift"] > 1.5])
    return {
        "lift_promedio":       round(lift_prom, 2),
        "indice_afinidad":     min(round((lift_prom / 3.0) * 100, 1), 100.0),
        "potencial_ventas":    round(conf_prom * 100, 1),
        "reduccion_recorrido": min(round(fuertes * 2.5, 1), 50.0),
        "ticket_extra":        round(conf_prom * 15, 2),
        "n_reglas":            len(reglas),
        "n_reglas_fuertes":    fuertes,
    }


def obtener_rol(email: str, id_usuario: str) -> str:
    if not email:
        return "analista"
    try:
        res = supabase.table("usuarios_perfiles").select("rol").eq("email", email).execute()
        if res.data:
            return res.data[0]["rol"]
        if id_usuario:
            supabase.table("usuarios_perfiles").insert({
                "id": id_usuario, "email": email, "rol": "analista"
            }).execute()
        return "analista"
    except Exception as e:
        st.sidebar.error(f"🚨 Error BD: {e}")
        return "analista"


@st.dialog("⚠️ Confirmar cierre de sesión")
def confirmar_logout():
    st.warning("¿Estás seguro de que deseas cerrar tu sesión?")
    st.write("Tendrás que volver a iniciar sesión para acceder al sistema.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Paso 1: Desvincular credenciales")
        logout_button(url=_url, apiKey=_key)
    with col2:
        st.caption("Paso 2: Confirmar salida")
        if st.button("✅ Confirmar y salir", use_container_width=True):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════════════════════════
if st.session_state["user"] is None:

    # Logo centrado
    st.markdown("""
    <div class="login-shell">
        <div class="login-logo">
            <div class="login-logo-icon">🛒</div>
            <h1>OPSO</h1>
            <p>Optimal Placement Stock</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        tab_login, tab_signup, tab_google = st.tabs([
            "Iniciar sesión",
            "Crear cuenta",
            "Google",
        ])

        # ── Tab 1: Iniciar sesión con email/contraseña ──────────────────
        with tab_login:
            with st.form("form_login", border=False):
                email_login = st.text_input(
                    "Correo electrónico",
                    placeholder="nombre@utp.ac.pa",
                )
                pass_login = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="••••••••",
                )
                if st.form_submit_button("Iniciar sesión", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({
                            "email": email_login,
                            "password": pass_login,
                        })
                        st.session_state["user"] = (
                            {"user": res.user.model_dump()} if res.user else res
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"🚨 No se pudo iniciar sesión: {e}")

        # ── Tab 2: Crear cuenta ─────────────────────────────────────────
        with tab_signup:
            with st.form("form_signup", border=False):
                nombre_signup = st.text_input(
                    "Nombre completo",
                    placeholder="Tu nombre completo",
                )
                email_signup = st.text_input(
                    "Correo electrónico",
                    placeholder="nombre@utp.ac.pa",
                    key="su_email",
                )
                pass_signup = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="Mínimo 8 caracteres",
                    key="su_pass",
                )
                if st.form_submit_button("Crear cuenta", use_container_width=True):
                    if len(pass_signup) < 8:
                        st.error("⚠️ La contraseña debe tener al menos 8 caracteres.")
                    else:
                        try:
                            supabase.auth.sign_up({
                                "email": email_signup,
                                "password": pass_signup,
                                "options": {"data": {"full_name": nombre_signup}},
                            })
                            st.success("✅ Cuenta creada. Revisa tu correo para confirmarla.")
                        except Exception as e:
                            st.error(f"🚨 No se pudo crear la cuenta: {e}")

        # ── Tab 3: Google OAuth (única instancia) ───────────────────────
        with tab_google:
            st.markdown("""
            <div class="google-tab-hint">
                <p>Inicia sesión o crea tu cuenta usando Google.<br>
                Serás redirigido a la página de autorización.</p>
            </div>
            """, unsafe_allow_html=True)

            gi = login_form(url=_url, apiKey=_key, providers=["google"])
            if gi:
                st.session_state["user"] = gi
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
else:
    usuario_data  = st.session_state["user"]
    id_usuario    = ""
    email_usuario = ""

    if isinstance(usuario_data, dict):
        inner = usuario_data.get("user", usuario_data)
        email_usuario = inner.get("email", "")
        id_usuario    = inner.get("id", "")
    else:
        user_obj      = getattr(usuario_data, "user", None)
        base          = user_obj if user_obj else usuario_data
        email_usuario = getattr(base, "email", "")
        id_usuario    = getattr(base, "id", "")

    rol_usuario = obtener_rol(email_usuario, id_usuario)

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.title("Menú OPSO")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=80)

    if rol_usuario == "admin":
        menu = ["Página Principal", "Carga de datos", "Análisis de Patrones",
                "Simulación de Layout", "Reportes", "Panel Gerencial", "Gestión de Usuarios"]
    elif rol_usuario == "gerente":
        menu = ["Página Principal", "Reportes", "Panel Gerencial"]
    else:
        menu = ["Página Principal", "Análisis de Patrones", "Simulación de Layout"]

    eleccion = st.sidebar.radio("Navegación", menu)
    st.sidebar.markdown("---")
    st.sidebar.info(f"Usuario: {email_usuario}\nRol: {rol_usuario.upper()}")

    if st.sidebar.button("🚪 Cerrar sesión", use_container_width=True):
        confirmar_logout()

    # ════════════════════════════════════════════════════════════════════
    # 1. PÁGINA PRINCIPAL
    # ════════════════════════════════════════════════════════════════════
    if eleccion == "Página Principal":
        st.title("🛒 OPSO - Optimal Placement Stock")
        st.markdown("### Optimización de distribución de supermercados mediante Machine Learning")
        st.write("""
        Bienvenido al sistema OPSO. Esta herramienta analiza patrones reales de compra
        utilizando el algoritmo Apriori y genera recomendaciones estratégicas para
        reorganizar los pasillos y productos de su establecimiento.
        """)
        st.info("**Objetivo:** Minimizar recorridos innecesarios, mejorar la experiencia del cliente y maximizar las ventas cruzadas.")

    # ════════════════════════════════════════════════════════════════════
    # 2. CARGA DE DATOS
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Carga de datos":
        import random

        st.title("📂 Entrada y Preprocesamiento de Datos")
        st.write("Sube el archivo CSV con las transacciones del supermercado o conéctate a la base de datos.")

        # ── Fuente 1: CSV local ───────────────────────────────────────
        archivo_subido = st.file_uploader("Cargar archivo CSV", type=["csv"])
        if archivo_subido is not None:
            df = pd.read_csv(archivo_subido)
            cols_req = {"ID_Factura", "Producto"}
            if not cols_req.issubset(df.columns):
                faltantes = cols_req - set(df.columns)
                st.error(
                    f"❌ Columnas requeridas faltantes: **{', '.join(faltantes)}**.\n\n"
                    f"Columnas encontradas: {', '.join(df.columns.tolist())}."
                )
            else:
                st.session_state["df_bruto"] = df

        st.markdown("---")

        # ── Fuente 2: Supabase + Generador sintético ──────────────────
        col_src1, col_src2 = st.columns(2)

        with col_src1:
            st.markdown("**☁️ Sincronizar con Supabase**")
            st.caption("Descarga las transacciones reales almacenadas en la nube.")
            if st.button("🔄 Descargar datos desde Supabase", use_container_width=True):
                with st.spinner("Conectando con la base de datos..."):
                    try:
                        res = supabase.table("transacciones").select("*").execute()
                        if res.data:
                            df_nube = pd.DataFrame(res.data).rename(
                                columns={"id_factura": "ID_Factura", "producto": "Producto"})
                            st.session_state["df_bruto"] = df_nube
                            st.success(f"¡Se descargaron {len(df_nube)} registros!")
                        else:
                            st.warning("La base de datos está vacía.")
                    except Exception as e:
                        st.error(f"Error al conectar con Supabase: {e}")

        with col_src2:
            st.markdown("**🧪 Generar datos sintéticos**")
            st.caption("Crea facturas realistas y súbelas automáticamente a Supabase.")
            n_facturas = st.number_input(
                "Número de facturas a generar", min_value=10, max_value=2000,
                value=200, step=50, key="n_sint"
            )

        # ── Generador sintético basado en perfiles de cliente ─────────
        PERFILES = {
            "Familiar": {
                "productos": [
                    "Leche", "Pan molde", "Huevos", "Arroz", "Frijoles",
                    "Pollo entero", "Papel Higiénico", "Detergente",
                    "Queso amarillo", "Aceite vegetal", "Azúcar", "Sal",
                    "Salsa de tomate", "Pasta espagueti", "Jabón de lavar",
                ],
                "peso": 0.35,
            },
            "Parrillero_FinDeSemana": {
                "productos": [
                    "Carne de res", "Chorizo", "Carbón", "Cerveza",
                    "Snacks", "Hielo", "Salsa BBQ", "Chuleta de cerdo",
                    "Salchicha", "Servilletas", "Desechables (Platos)",
                    "Papel Aluminio", "Maíz para asar", "Salsa Tabasco",
                ],
                "peso": 0.25,
            },
            "Estudiante_Rapido": {
                "productos": [
                    "Sopa instantánea", "Soda", "Galletas", "Pollo frito",
                    "Café", "Cereal", "Comida congelada", "Atún en lata",
                    "Pan de molde", "Mayonesa", "Bebida Energética",
                    "Chocolates", "Agua embotellada",
                ],
                "peso": 0.25,
            },
            "Saludable": {
                "productos": [
                    "Pechuga de pollo", "Avena", "Yogur Griego",
                    "Manzanas", "Espinaca", "Atún", "Té verde",
                    "Aceite de oliva", "Almendras", "Granola",
                    "Leche deslactosada", "Pasta integral", "Quinoa",
                ],
                "peso": 0.15,
            },
        }

        # Productos de compra impulsiva cerca de la caja
        PRODUCTOS_CAJA = [
            "Goma de mascar", "Baterías", "Agua embotellada",
            "Chocolates", "Chiclets", "Encendedor",
        ]

        def generar_factura(id_factura: str) -> list[dict]:
            """
            Genera una factura siguiendo el patrón de perfiles de cliente:
            - Elige un perfil ponderado por su frecuencia real.
            - Selecciona 3-6 productos de ese perfil.
            - 25% de probabilidad de agregar un ítem de compra impulsiva en caja.
            """
            perfiles_keys = list(PERFILES.keys())
            pesos = [PERFILES[p]["peso"] for p in perfiles_keys]
            perfil_actual = random.choices(perfiles_keys, weights=pesos, k=1)[0]

            pool = PERFILES[perfil_actual]["productos"]
            n_prods = random.randint(3, min(6, len(pool)))
            productos_comprados = random.sample(pool, n_prods)

            # Compra impulsiva cerca de la caja (25 % de probabilidad)
            if random.random() < 0.25:
                productos_comprados.append(random.choice(PRODUCTOS_CAJA))

            return [{"id_factura": id_factura, "producto": prod}
                    for prod in productos_comprados]

        # ── Botón de generación ───────────────────────────────────────
        if st.button("🧪 Generar y subir datos sintéticos a Supabase",
                     use_container_width=True, key="btn_sintetico"):

            n_fact = int(n_facturas)
            registros_total = []
            # Prefijo F- + offset aleatorio para evitar colisiones con datos reales
            base_id = random.randint(1_000, 9_000)

            with st.spinner(f"Generando {n_fact} facturas por perfiles de cliente..."):
                for i in range(1, n_fact + 1):
                    id_fact = f"F-{base_id + i:05d}"
                    registros_total.extend(generar_factura(id_fact))

            # ── Estadísticas del lote generado ───────────────────────
            df_preview = pd.DataFrame(registros_total).rename(
                columns={"id_factura": "ID_Factura", "producto": "Producto"})

            # Conteo por perfil (inferido por productos característicos)
            total_lineas = len(registros_total)
            prom_prods = total_lineas // n_fact

            st.info(
                f"✅ Se generaron **{total_lineas} líneas** en **{n_fact} facturas** "
                f"· ~**{prom_prods} productos/factura** en promedio."
            )

            # Distribución de productos generados
            top_prods = df_preview["Producto"].value_counts().head(8)
            col_stat1, col_stat2 = st.columns([1, 1])
            with col_stat1:
                with st.expander("👁️ Vista previa (primeros 30 registros)"):
                    st.dataframe(df_preview.head(30), use_container_width=True)
            with col_stat2:
                with st.expander("📊 Top 8 productos generados"):
                    st.bar_chart(top_prods, color="#ff5b23")

            # Subir en lotes de 500 (límite recomendado de Supabase)
            LOTE = 500
            lotes = [registros_total[i:i + LOTE] for i in range(0, len(registros_total), LOTE)]
            prog = st.progress(0, text="Subiendo a Supabase…")
            errores = []

            for idx_lote, lote in enumerate(lotes):
                try:
                    supabase.table("transacciones").insert(lote).execute()
                except Exception as e:
                    errores.append(str(e))
                prog.progress(
                    (idx_lote + 1) / len(lotes),
                    text=f"Subiendo lote {idx_lote + 1} / {len(lotes)}…"
                )

            prog.empty()

            if errores:
                st.warning(
                    f"⚠️ Se subieron {len(lotes) - len(errores)}/{len(lotes)} lotes. "
                    f"Errores: {'; '.join(errores[:3])}"
                )
            else:
                st.success(
                    f"🎉 ¡{len(registros_total)} registros subidos exitosamente a Supabase! "
                    "Usa el botón **Descargar datos desde Supabase** para cargarlos."
                )

        st.markdown("---")

        # ── Datos en memoria ──────────────────────────────────────────
        if st.session_state["df_bruto"] is not None:
            df = st.session_state["df_bruto"]
            st.success("¡Datos listos en memoria!")

            st.markdown("### 📊 Top 10 Productos Más Vendidos")
            st.bar_chart(df["Producto"].value_counts().head(10), color="#ff5b23")

            st.write("Vista previa de las transacciones:")
            st.dataframe(df.head())

            if st.button("Ejecutar Preprocesamiento (Crear Matriz)"):
                with st.spinner("Transformando datos..."):
                    cesta = pd.crosstab(df["ID_Factura"], df["Producto"]) > 0
                    st.session_state["df_cesta"] = cesta
                st.success("¡Datos transformados! Listos para el análisis matemático.")
                st.dataframe(st.session_state["df_cesta"].head())

    # ════════════════════════════════════════════════════════════════════
    # 3. ANÁLISIS DE PATRONES
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Análisis de Patrones":
        st.title("🧠 Análisis de Patrones (Algoritmo Apriori)")

        if st.session_state["df_cesta"] is None:
            st.warning("⚠️ Primero ve a 'Carga de datos' y preprocesa tu información.")
        else:
            # ── Controles manuales ────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                min_soporte = st.slider("Soporte Mínimo (%)", 1, 50, 5) / 100.0
                st.caption("Ej: 0.05 → productos en al menos el 5% de las facturas.")
            with col2:
                min_confianza = st.slider("Confianza Mínima (%)", 10, 100, 50) / 100.0
                st.caption("Ej: 0.50 → al comprar A, 50% de probabilidad de llevar B.")

            col_b1, col_b2 = st.columns(2)

            # ── Botón manual ──────────────────────────────────────────
            with col_b1:
                if st.button("▶ Ejecutar Algoritmo Apriori", use_container_width=True):
                    with st.spinner("Buscando patrones frecuentes..."):
                        itemsets = apriori(st.session_state["df_cesta"],
                                           min_support=min_soporte, use_colnames=True)
                        if itemsets.empty:
                            st.error("Sin patrones con estos parámetros. Baja el soporte.")
                        else:
                            reglas = association_rules(itemsets, metric="confidence",
                                                       min_threshold=min_confianza)
                            if reglas.empty:
                                st.error("Sin reglas con esa confianza. Baja la confianza.")
                            else:
                                reglas["antecedents"] = reglas["antecedents"].apply(
                                    lambda x: ", ".join(list(x)))
                                reglas["consequents"] = reglas["consequents"].apply(
                                    lambda x: ", ".join(list(x)))
                                st.session_state["reglas"] = reglas
                                st.success(f"¡Se encontraron {len(reglas)} reglas de asociación!")

            # ── Botón auto-entrenamiento ──────────────────────────────
            with col_b2:
                if st.button("🤖 Auto-entrenar (llenar estantes)", use_container_width=True):
                    N_ESTANTES = 5
                    cesta = st.session_state["df_cesta"]
                    mejor_reglas = None
                    mejor_params = None
                    log_msgs = []

                    # Búsqueda de grilla: soporte decreciente × confianza decreciente
                    soportes   = [0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
                    confianzas = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]

                    progress = st.progress(0, text="Iniciando búsqueda de parámetros óptimos…")
                    total_combos = len(soportes) * len(confianzas)
                    idx = 0

                    for sp in soportes:
                        for cf in confianzas:
                            idx += 1
                            progress.progress(
                                idx / total_combos,
                                text=f"Probando soporte={sp:.0%} · confianza={cf:.0%}…"
                            )
                            try:
                                its = apriori(cesta, min_support=sp, use_colnames=True)
                                if its.empty:
                                    continue
                                rls = association_rules(its, metric="confidence", min_threshold=cf)
                                if rls.empty:
                                    continue

                                # Contar pares únicos disponibles (= estantes llenables)
                                pares = set()
                                for _, r in rls.iterrows():
                                    pares.add(frozenset([
                                        ", ".join(list(r["antecedents"])) if hasattr(r["antecedents"], "__iter__") and not isinstance(r["antecedents"], str) else r["antecedents"],
                                        ", ".join(list(r["consequents"])) if hasattr(r["consequents"], "__iter__") and not isinstance(r["consequents"], str) else r["consequents"],
                                    ]))

                                log_msgs.append(
                                    f"sp={sp:.0%} cf={cf:.0%} → {len(rls)} reglas, {len(pares)} pares únicos"
                                )

                                # Guardamos si supera el mínimo y mejora el lift promedio
                                lift_avg = rls["lift"].mean()
                                if len(pares) >= N_ESTANTES:
                                    if mejor_reglas is None or lift_avg > mejor_reglas["lift"].mean():
                                        mejor_reglas = rls.copy()
                                        mejor_params = (sp, cf, len(pares), round(lift_avg, 3))

                            except Exception:
                                continue

                    progress.empty()

                    if mejor_reglas is not None:
                        mejor_reglas["antecedents"] = mejor_reglas["antecedents"].apply(
                            lambda x: ", ".join(list(x)) if not isinstance(x, str) else x)
                        mejor_reglas["consequents"] = mejor_reglas["consequents"].apply(
                            lambda x: ", ".join(list(x)) if not isinstance(x, str) else x)
                        st.session_state["reglas"] = mejor_reglas

                        sp, cf, n_pares, lift_avg = mejor_params
                        st.success(
                            f"✅ **Parámetros óptimos encontrados:** "
                            f"Soporte = **{sp:.0%}** · Confianza = **{cf:.0%}**  \n"
                            f"→ {len(mejor_reglas)} reglas · {n_pares} pares únicos · "
                            f"Lift promedio = **{lift_avg}**"
                        )

                        with st.expander("📋 Ver log de búsqueda"):
                            for msg in log_msgs:
                                st.text(msg)
                    else:
                        st.error(
                            "❌ No se encontró ninguna combinación de parámetros que genere "
                            f"suficientes reglas para llenar {N_ESTANTES} estantes. "
                            "Verifica que el dataset tenga suficientes transacciones."
                        )
                        with st.expander("📋 Ver log de búsqueda"):
                            for msg in log_msgs:
                                st.text(msg)

            # ── Visualización de resultados ───────────────────────────
            if st.session_state["reglas"] is not None:
                rules = st.session_state["reglas"]

                st.markdown("---")
                st.markdown("### 🔍 Buscador de Asociaciones")
                filtro = st.text_input("Filtrar por producto (ej. Cerveza, Pan, Leche):", "")
                if filtro:
                    mask = (rules["antecedents"].str.contains(filtro, case=False, na=False) |
                            rules["consequents"].str.contains(filtro, case=False, na=False))
                    reglas_vis = rules[mask]
                else:
                    reglas_vis = rules

                tabla = reglas_vis[["antecedents", "consequents", "support", "confidence", "lift"]]\
                    .sort_values("lift", ascending=False).copy()
                tabla.columns = ["Si compran (Antecedente)", "También compran (Consecuente)",
                                  "Soporte", "Confianza", "Lift (Fuerza)"]

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("### 📈 Soporte vs Confianza")
                    st.info("El tamaño de la burbuja representa el Lift.")
                    if not tabla.empty:
                        st.scatter_chart(data=reglas_vis, x="support", y="confidence",
                                         size="lift", color="#ff5b23")
                    else:
                        st.warning("No hay datos para graficar con ese filtro.")

                with col_g2:
                    st.markdown("### 🕸️ Red de Compras Frecuentes")
                    st.info("Conexiones entre productos asociados.")
                    if not tabla.empty:
                        G = nx.DiGraph()
                        for _, row in tabla.head(20).iterrows():
                            G.add_edge(row["Si compran (Antecedente)"],
                                       row["También compran (Consecuente)"],
                                       weight=row["Lift (Fuerza)"])
                        fig_net, ax_net = plt.subplots(figsize=(6, 6))
                        fig_net.patch.set_facecolor("#141414")
                        ax_net.set_facecolor("#141414")
                        nx.draw(G, nx.spring_layout(G, k=0.8, seed=42),
                                with_labels=True, node_color="#ff5b23",
                                node_size=2000, font_size=9, font_weight="bold",
                                font_color="#ffffff", edge_color="#18cccd",
                                ax=ax_net, arrows=True, arrowsize=15, alpha=0.9)
                        st.pyplot(fig_net)
                    else:
                        st.warning("No hay conexiones para graficar.")

                st.markdown("### 📋 Resultados Detallados")
                st.dataframe(tabla, use_container_width=True)
                st.download_button(
                    label="📥 Descargar Reglas (CSV)",
                    data=tabla.to_csv(index=False).encode("utf-8"),
                    file_name="OPSO_Reglas_Apriori.csv",
                    mime="text/csv",
                )

    # ════════════════════════════════════════════════════════════════════
    # 4. SIMULACIÓN DE LAYOUT
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Simulación de Layout":
        st.title("🗺️ Simulación y Optimización del Layout")

        # ── Colores por pasillo ───────────────────────────────────────
        COLORES = ["#ff5b23", "#18cccd", "#fbbf24", "#a78bfa", "#4ade80"]

        # ── Coordenadas fijas de 5 estantes en el plano ──────────────
        # Cada entrada: [x, y, ancho, alto]
        COORDS = [
            [0.8, 7.8, 3.8, 1.4],   # superior izq
            [5.4, 7.8, 3.8, 1.4],   # superior der
            [0.8, 4.2, 2.6, 3.0],   # central izq
            [4.0, 4.2, 2.6, 3.0],   # central mid
            [6.6, 4.2, 2.6, 3.0],   # central der
        ]

        # ── Helper: nombre inteligente para un grupo de productos ─────
        CATEGORIAS = {
            "parrilla":   ["carne", "cerdo", "res", "pollo", "cerveza", "carbón", "carbón",
                           "bbq", "salsa", "chorizo", "embutido"],
            "desayuno":   ["pan", "huevo", "leche", "queso", "mantequilla", "café",
                           "cereal", "avena", "jamón", "yogur"],
            "bebidas":    ["soda", "refresco", "agua", "jugo", "cerveza", "vino",
                           "bebida", "gaseosa", "té", "energética"],
            "snacks":     ["snack", "chips", "galleta", "chocolate", "dulce", "caramelo",
                           "maní", "palomita", "barra"],
            "limpieza":   ["detergente", "jabón", "cloro", "papel", "toalla", "escoba",
                           "desinfectante", "límpido", "suavizante"],
            "abarrotes":  ["arroz", "frijol", "fideo", "pasta", "sopa", "atún",
                           "sardina", "lata", "harina", "aceite", "sal", "azúcar"],
            "frutas":     ["fruta", "verdura", "vegetal", "tomate", "cebolla", "ajo",
                           "papaya", "mango", "piña", "plátano"],
            "higiene":    ["shampoo", "cepillo", "pasta dental", "desodorante",
                           "crema", "loción", "pañal", "toalla sanitaria"],
        }

        EMOJIS = {
            "parrilla": "🔥", "desayuno": "🍳", "bebidas": "🥤",
            "snacks": "🍿", "limpieza": "🧹", "abarrotes": "🛒",
            "frutas": "🥦", "higiene": "🧴", "general": "📦",
        }

        def nombre_pasillo(productos: list[str]) -> tuple[str, str]:
            """Devuelve (emoji + nombre corto, nombre completo descriptivo)."""
            texto = " ".join(productos).lower()
            scores = {cat: 0 for cat in CATEGORIAS}
            for cat, keywords in CATEGORIAS.items():
                for kw in keywords:
                    if kw in texto:
                        scores[cat] += 1
            mejor = max(scores, key=scores.get)
            if scores[mejor] == 0:
                mejor = "general"
            emoji = EMOJIS.get(mejor, "📦")
            nombres_cap = {
                "parrilla": "Zona Parrilla", "desayuno": "Zona Desayuno",
                "bebidas": "Bebidas", "snacks": "Snacks & Dulces",
                "limpieza": "Limpieza", "abarrotes": "Abarrotes",
                "frutas": "Frutas & Verduras", "higiene": "Higiene Personal",
                "general": "Pasillo General",
            }
            return f"{emoji} {nombres_cap[mejor]}", mejor

        # ── Construir grupos de productos por pasillo (OPSO) ─────────
        # Cada grupo = set de productos que van juntos
        grupos_opso: list[set] = []      # lista de sets de productos
        nombres_opso: list[str] = []     # nombre corto del pasillo
        mapeados: set = set()

        if st.session_state["reglas"] is not None and not st.session_state["reglas"].empty:
            col_lift = "lift" if "lift" in st.session_state["reglas"].columns else "Lift (Fuerza)"
            reglas_sorted = st.session_state["reglas"].sort_values(col_lift, ascending=False)

            for _, row in reglas_sorted.iterrows():
                if len(grupos_opso) >= 5:
                    break
                ant = row["antecedents"]
                con = row["consequents"]
                par = frozenset([ant, con])
                if par in mapeados:
                    continue
                mapeados.add(par)

                # Productos del antecedente y consecuente
                prods_ant = [p.strip() for p in ant.split(",")]
                prods_con = [p.strip() for p in con.split(",")]
                grupo = set(prods_ant + prods_con)

                # Intentar fusionar con un grupo existente si comparte productos
                fusionado = False
                for g in grupos_opso:
                    if g & grupo:  # intersección
                        g.update(grupo)
                        fusionado = True
                        break
                if not fusionado and len(grupos_opso) < 5:
                    grupos_opso.append(grupo)

        # Rellenar slots vacíos con productos más frecuentes no asignados
        if st.session_state["df_bruto"] is not None:
            productos_frecuentes = st.session_state["df_bruto"]["Producto"].value_counts().index.tolist()
            ya_asignados = {p for g in grupos_opso for p in g}
            for prod in productos_frecuentes:
                if len(grupos_opso) >= 5:
                    break
                if prod not in ya_asignados:
                    grupos_opso.append({prod})
                    ya_asignados.add(prod)

        # Fallback si no hay datos
        if not grupos_opso:
            grupos_opso = [
                {"Carnes", "Cerveza", "Carbón", "Salsa BBQ"},
                {"Sopas", "Soda", "Snacks"},
                {"Pan", "Huevos", "Leche", "Queso"},
                {"Arroz", "Frijoles", "Atún"},
                {"Papel Higiénico", "Detergente"},
            ]

        # Generar nombres inteligentes
        nombres_opso = []
        for g in grupos_opso:
            nombre, _ = nombre_pasillo(list(g))
            nombres_opso.append(nombre)

        # ── Plano matplotlib: solo muestra número + nombre corto ──────
        def dibujar_plano_v2(estantes_data, indices_ruta=None):
            """
            estantes_data: lista de (x, y, w, h, label_corto, color)
            Dibuja sin texto superpuesto — solo un número de pasillo
            grande y el nombre corto en dos líneas máximo.
            """
            fig, ax = plt.subplots(figsize=(7, 5.5))
            fig.patch.set_facecolor("#0a0a0a")
            ax.set_facecolor("#0a0a0a")
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")

            # Cajas fijas: Entrada y Cajas
            for rx, ry, rw, rh, rtxt, rclr in [
                (0.5, 0.5, 2.5, 1.5, "CAJAS", "#2a2a2a"),
                (7.0, 0.5, 2.5, 1.5, "ENTRADA", "#1a3a1a"),
            ]:
                ax.add_patch(patches.FancyBboxPatch(
                    (rx, ry), rw, rh,
                    boxstyle="round,pad=0.05",
                    facecolor=rclr, edgecolor="#444", linewidth=1.2,
                ))
                ax.text(rx + rw / 2, ry + rh / 2, rtxt,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="#aaaaaa")

            centros_x, centros_y = [], []
            for i, (x, y, w, h, label, color) in enumerate(estantes_data):
                en_ruta = indices_ruta and i in indices_ruta
                # Borde naranja/cian si está en ruta
                edge_color = "#ff5b23" if en_ruta else "#333333"
                edge_lw    = 2.5 if en_ruta else 1.0

                ax.add_patch(patches.FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0.08",
                    facecolor=color + "22",   # muy transparente
                    edgecolor=edge_color,
                    linewidth=edge_lw,
                ))

                # Número grande centrado
                cx, cy = x + w / 2, y + h / 2
                ax.text(cx, cy + 0.18, str(i + 1),
                        ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color=color, alpha=0.85)

                # Nombre en dos líneas debajo del número
                palabras = label.split()
                mid = len(palabras) // 2 or 1
                linea1 = " ".join(palabras[:mid])
                linea2 = " ".join(palabras[mid:])
                ax.text(cx, cy - 0.38,
                        f"{linea1}\n{linea2}" if linea2 else linea1,
                        ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="#e0e0e0", linespacing=1.3)

                centros_x.append(cx)
                centros_y.append(cy)

            # Ruta punteada
            if indices_ruta and len(indices_ruta) > 0:
                rx_path = [8.25] + [centros_x[i] for i in indices_ruta] + [1.75]
                ry_path = [1.25] + [centros_y[i] for i in indices_ruta] + [1.25]
                ax.plot(rx_path, ry_path,
                        color="#18cccd", linestyle="--",
                        linewidth=2.0, marker="o", markersize=5, alpha=0.9)

            fig.tight_layout(pad=0.3)
            return fig

        # ── Preparar datos para los dos planos ───────────────────────
        # Plano OPSO
        estantes_opso_plot = [
            [COORDS[i][0], COORDS[i][1], COORDS[i][2], COORDS[i][3],
             nombres_opso[i], COLORES[i]]
            for i in range(min(len(grupos_opso), 5))
        ]

        # Plano tradicional: un producto por pasillo (el más frecuente del grupo)
        estantes_trad_plot = []
        if st.session_state["df_bruto"] is not None:
            prods_freq = st.session_state["df_bruto"]["Producto"].value_counts().index.tolist()
            for i, c in enumerate(COORDS):
                label = f"📦 {prods_freq[i]}" if i < len(prods_freq) else f"Vacío {i+1}"
                estantes_trad_plot.append([c[0], c[1], c[2], c[3], label, "#555555"])
        else:
            fallback = ["🥩 Carnes", "🥛 Lácteos", "🍞 Panadería", "🥤 Bebidas", "📦 Varios"]
            estantes_trad_plot = [
                [COORDS[i][0], COORDS[i][1], COORDS[i][2], COORDS[i][3],
                 fallback[i], "#555555"] for i in range(5)
            ]

        # ── Simulador de ruta ─────────────────────────────────────────
        st.markdown("### 🚶 Simulador de Recorrido")
        opciones_ruta = [f"Pasillo {i+1} — {nombres_opso[i]}" for i in range(len(grupos_opso))]
        seleccion = st.multiselect("Seleccione los pasillos que visitará el cliente:", opciones_ruta)
        indices_ruta = [opciones_ruta.index(z) for z in seleccion]

        # ── Planos lado a lado ────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Layout Tradicional** — por categoría genérica")
            st.pyplot(dibujar_plano_v2(estantes_trad_plot, []))
        with col2:
            st.markdown("**Layout Optimizado OPSO** — por comportamiento de compra")
            st.pyplot(dibujar_plano_v2(estantes_opso_plot, indices_ruta))

        # ── Listas detalladas de productos por pasillo ────────────────
        st.markdown("---")
        st.markdown("### 🗂️ Distribución de Productos por Pasillo (OPSO)")
        st.caption("Cada pasillo agrupa productos con alta afinidad de compra según el modelo Apriori.")

        cols = st.columns(min(len(grupos_opso), 3))
        for i, (grupo, nombre) in enumerate(zip(grupos_opso, nombres_opso)):
            with cols[i % 3]:
                color_hex = COLORES[i]
                productos_lista = sorted(grupo)

                st.markdown(
                    f"""<div style="
                        background: {color_hex}18;
                        border: 1px solid {color_hex}55;
                        border-top: 4px solid {color_hex};
                        border-radius: 10px;
                        padding: 14px 16px;
                        margin-bottom: 12px;
                    ">
                    <p style="font-size:13px;font-weight:700;color:{color_hex};
                               margin:0 0 8px;letter-spacing:0.02em;">
                        {nombre} &nbsp;·&nbsp; Pasillo {i+1}
                    </p>
                    <ul style="margin:0;padding-left:16px;list-style:disc;">
                        {"".join(f'<li style="font-size:13px;color:var(--c-text,#f5f5f5);line-height:1.7;">{p}</li>' for p in productos_lista)}
                    </ul>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Enviar al panel gerencial ─────────────────────────────────
        st.markdown("---")
        if st.button("💾 Enviar Propuesta al Panel Gerencial"):
            try:
                supabase.table("layouts_historial").insert({
                    "nombre_layout": f"Propuesta OPSO - {datetime.date.today()}",
                    "asociaciones": str([f"{nombres_opso[i]}: {', '.join(sorted(grupos_opso[i]))}"
                                         for i in range(len(grupos_opso))]),
                    "creado_por": email_usuario,
                    "estado": "Pendiente",
                }).execute()
                st.success("¡Propuesta enviada! El Gerente puede revisarla en su panel.")
            except Exception as e:
                st.error(f"Error al guardar propuesta: {e}")

    # ════════════════════════════════════════════════════════════════════
    # 5. REPORTES
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Reportes":
        st.title("📊 Salidas del Sistema y Reportes Gerenciales")

        reglas = st.session_state.get("reglas")

        if reglas is not None and not reglas.empty:
            kpis = calcular_kpis(reglas)
            st.write(f"Métricas basadas en **{kpis['n_reglas']} reglas** encontradas por OPSO.")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Reducción Recorrido (est.)", f"{kpis['reduccion_recorrido']} %",
                          delta=f"{kpis['n_reglas_fuertes']} reglas lift > 1.5")
            with k2:
                st.metric("Potencial Ventas Cruzadas", f"{kpis['potencial_ventas']} %",
                          delta=f"+ B/. {kpis['ticket_extra']:.2f} por ticket")
            with k3:
                st.metric("Índice Afinidad Layout", f"{kpis['indice_afinidad']} / 100",
                          delta=f"Lift prom: {kpis['lift_promedio']}")
        else:
            st.warning("⚠️ Ejecuta el algoritmo Apriori en 'Análisis de Patrones' para ver KPIs reales.")
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Reducción Recorrido", "— %", delta="Sin datos aún")
            with k2: st.metric("Ventas Cruzadas", "— %", delta="Sin datos aún")
            with k3: st.metric("Índice Afinidad", "— / 100", delta="Sin datos aún")

        # ROI Calculator
        st.markdown("### 💰 Calculadora de ROI")
        st.info("Estima la ganancia económica al aplicar las reglas de OPSO.")
        col_r1, col_r2 = st.columns(2)
        ticket_prom   = col_r1.number_input("Ticket Promedio (B/.)", value=25.0, step=1.0)
        facturas_mes  = col_r2.number_input("Facturas mensuales", value=1500, step=100)

        if reglas is not None and not reglas.empty:
            col_lift = "lift" if "lift" in reglas.columns else "Lift (Fuerza)"
            lift_avg = reglas[col_lift].mean()
            mejora   = max((lift_avg - 1) * 0.05, 0.02)
            proyeccion = ticket_prom * facturas_mes * mejora
            st.success(
                f"**Proyección:** Con Lift promedio {lift_avg:.2f}, estimamos un incremento de "
                f"**B/. {proyeccion:,.2f} mensuales**."
            )
        else:
            st.warning("Ejecuta Apriori para calcular la proyección financiera.")

        # Matriz de movimientos
        st.markdown("### 📋 Matriz de Planificación de Movimientos")
        df_reporte = pd.DataFrame({
            "Zona Destino": ["Zona Parrilla", "Zona Estudiante", "Zona Desayuno", "Abarrotes", "Limpieza"],
            "Categorías Integradas": [
                "Carnes, Cerveza, Carbón, Salsa BBQ", "Sopa instantánea, Soda, Snacks",
                "Pan, Huevos, Leche, Queso", "Arroz, Frijoles, Atún", "Papel Higiénico, Detergente",
            ],
            "Justificación de Regla": [
                "Afinidad en fines de semana", "Compra rápida / conveniencia",
                "Alta correlación diaria", "Productos base estables", "Baja correlación con alimentos",
            ],
            "Prioridad": ["ALTA", "MEDIA", "ALTA", "BAJA", "MEDIA"],
        })
        st.dataframe(df_reporte, use_container_width=True)

        # Descargas
        st.markdown("### 📥 Descarga de Entregables")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.download_button(
                "💾 Descargar Matriz (CSV)",
                data=df_reporte.to_csv(index=False).encode("utf-8"),
                file_name="OPSO_Matriz_Movimientos.csv",
                mime="text/csv",
            )
        with col_b2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Reporte Ejecutivo OPSO", ln=True, align="C")
            pdf.set_font("Arial", "B", 12)
            pdf.ln(10)
            pdf.cell(0, 10, "Resumen de Optimizacion de Layout", ln=True, align="L")
            pdf.set_font("Arial", "", 11)

            if reglas is not None and not reglas.empty:
                kpis = calcular_kpis(reglas)
                cuerpo = (
                    f"Generado por OPSO. Se detectaron {kpis['n_reglas']} reglas de asociacion "
                    f"con lift promedio {kpis['lift_promedio']}. Reduccion estimada de recorrido: "
                    f"{kpis['reduccion_recorrido']}%. Potencial ventas cruzadas: {kpis['potencial_ventas']}%."
                )
            else:
                cuerpo = ("Generado por OPSO. No se detectaron reglas al momento de la exportacion. "
                          "Ejecute el algoritmo Apriori para obtener metricas reales.")

            pdf.multi_cell(0, 8, cuerpo)
            pdf_bytes = pdf.output(dest="S")
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode("latin-1")

            st.download_button(
                "📄 Descargar Reporte (PDF)",
                data=pdf_bytes,
                file_name="OPSO_Reporte_Ejecutivo.pdf",
                mime="application/pdf",
            )

    # ════════════════════════════════════════════════════════════════════
    # 6. PANEL GERENCIAL
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Panel Gerencial":
        st.title("🏢 Panel de Decisiones Gerenciales")
        st.write("Historial de propuestas de Layout para aprobación o rechazo.")

        try:
            res = supabase.table("layouts_historial").select("*").order("creado_en", desc=True).execute()
            if res.data:
                df_hist = pd.DataFrame(res.data)
                st.dataframe(df_hist[["creado_en", "nombre_layout", "creado_por", "estado"]],
                             use_container_width=True)

                st.markdown("### Auditar Propuesta")
                id_sel = st.selectbox("Seleccione el ID a evaluar:", df_hist["id"].tolist())
                fila   = df_hist[df_hist["id"] == id_sel].iloc[0]

                st.markdown("#### 🗂️ Distribución propuesta")
                try:
                    zonas = ast.literal_eval(fila["asociaciones"])
                    for i, zona in enumerate(zonas):
                        st.markdown(
                            f'<div class="estante-card"><b>Estante {i+1}:</b> {zona}</div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    st.write(fila["asociaciones"])

                st.markdown("---")
                col_ap, col_re = st.columns(2)
                with col_ap:
                    if st.button("✅ Aprobar Layout"):
                        supabase.table("layouts_historial").update({"estado": "Aprobado"}).eq("id", id_sel).execute()
                        st.success("Layout aprobado y registrado.")
                        st.rerun()
                with col_re:
                    if st.button("❌ Rechazar Propuesta"):
                        supabase.table("layouts_historial").update({"estado": "Rechazado"}).eq("id", id_sel).execute()
                        st.error("Propuesta rechazada.")
                        st.rerun()
            else:
                st.info("No hay propuestas de layout pendientes.")
        except Exception as e:
            st.error(f"Error cargando el historial: {e}. Verifica que la tabla 'layouts_historial' existe.")

    # ════════════════════════════════════════════════════════════════════
    # 7. GESTIÓN DE USUARIOS
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Gestión de Usuarios":
        st.title("👥 Panel de Gestión de Usuarios")
        st.write("Audita las cuentas registradas y reasigna roles en tiempo real.")

        try:
            res = supabase.table("usuarios_perfiles").select("*").execute()
            if res.data:
                df_u = pd.DataFrame(res.data)

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1: st.metric("Total Registrados", len(df_u))
                with col_m2: st.metric("Administradores", len(df_u[df_u["rol"] == "admin"]))
                with col_m3: st.metric("Analistas y Gerentes", len(df_u[df_u["rol"] != "admin"]))

                st.markdown("### 📋 Directorio de Cuentas")
                st.dataframe(df_u[["email", "rol"]], use_container_width=True)

                st.markdown("---")
                st.markdown("### ⚙️ Modificar Permisos")
                correo_sel = st.selectbox("Seleccione el correo a modificar:", df_u["email"].tolist())
                rol_actual = df_u[df_u["email"] == correo_sel]["rol"].values[0]
                roles      = ["admin", "gerente", "analista"]
                nuevo_rol  = st.selectbox("Nuevo rol:", roles, index=roles.index(rol_actual) if rol_actual in roles else 2)

                if st.button("Actualizar Privilegios"):
                    supabase.table("usuarios_perfiles").update({"rol": nuevo_rol}).eq("email", correo_sel).execute()
                    st.success(f"✅ {correo_sel} → {nuevo_rol.upper()}")
                    st.rerun()
            else:
                st.warning("La tabla 'usuarios_perfiles' no retornó registros.")
        except Exception as e:
            st.error(f"🚨 Error cargando panel de usuarios: {e}")