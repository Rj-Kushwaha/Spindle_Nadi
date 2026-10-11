"""Fleet monitoring dashboard for the Spindle Nadi prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.service import SpindleNadiService
from backend.esp32_serial import PHYSICAL_CHANNELS, Esp32SerialReader, list_esp32_ports
from backend.notifications import notifier_from_streamlit_secrets
from frontend.alert_messages import spoken_status_message
from frontend.audio_alerts import render_status_announcement
from data.generate_data import DEFAULT_OUTPUT, generate_dataset
from data.recorded_replay import (
    NUMERIC_CHANNELS,
    list_excel_sheets,
    list_supported_archive_members,
    parse_recorded_file,
    read_supported_archive_member,
    sensor_columns,
)
from ml.features import REQUIRED_SENSOR_COLUMNS
from ml.pipeline import SpindleNadiPipeline
from ml.prediction.risk_model import CRITICAL_THRESHOLD, WARNING_THRESHOLD
from simulator import FAULT_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_COLORS = {"NORMAL": "#38d39f", "WARNING": "#ffc857", "CRITICAL": "#ff647c"}
CHART_COLORS = {
    "temperature": "#ff8a65",
    "vibration": "#b69cff",
    "motor_current": "#4dc6d5",
    "rpm": "#62a5ff",
}
FAULT_LABELS = {
    "bearing_wear": "Bearing wear",
    "cooling_degradation": "Cooling degradation",
    "lubrication_loss": "Lubrication loss",
    "tool_wear": "Tool wear",
    "spindle_imbalance": "Spindle imbalance",
}
SENSOR_LABELS = {
    "temperature": "Temperature (°C)",
    "vibration": "Overall vibration (mm/s)",
    "vibration_x": "Vibration X (mm/s)",
    "vibration_y": "Vibration Y (mm/s)",
    "vibration_z": "Vibration Z (mm/s)",
    "motor_current": "Motor current (A)",
    "rpm": "Spindle speed (RPM)",
    "spindle_load_pct": "Spindle load (%)",
    "torque_nm": "Torque (N·m)",
    "bearing_temperature_c": "Bearing temperature (°C)",
    "coolant_flow_lpm": "Coolant flow (L/min)",
    "coolant_temperature_c": "Coolant temperature (°C)",
    "lubrication_pressure_bar": "Lubrication pressure (bar)",
    "acoustic_noise_db": "Acoustic level (dB)",
    "feed_rate_mm_min": "Feed rate (mm/min)",
    "tool_wear_pct_est": "Estimated tool wear (%)",
    "power_kw_est": "Estimated power (kW)",
}


@st.cache_resource
def load_pipeline() -> SpindleNadiPipeline:
    training_path = Path(DEFAULT_OUTPUT)
    if not training_path.exists():
        generate_dataset(training_path)
    training = pd.read_csv(training_path)
    if REQUIRED_SENSOR_COLUMNS.difference(training.columns):
        training = generate_dataset(training_path)
    return SpindleNadiPipeline(training)


def main() -> None:
    st.set_page_config(
        page_title="Spindle Nadi | Machine Health",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        .stApp { background: #0b1116; color: #e8eff2; font-family: 'Manrope', sans-serif; }
        [data-testid="stSidebar"] { background: #101a20; border-right: 1px solid #203039; }
        .block-container { padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1500px; }
        h1, h2, h3 { letter-spacing: -0.035em; }
        .hero { padding: 1.2rem 0 1rem; }
        .eyebrow { color: #72d7bd; font-size: .72rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
        .hero-title { font-size: 2.4rem; line-height: 1.1; font-weight: 800; margin: .25rem 0; }
        .muted { color: #95a8b1; }
        .status-pill { border: 1px solid currentColor; border-radius: 999px; padding: .3rem .7rem;
                       font-size: .72rem; font-weight: 800; letter-spacing: .08em; display: inline-block; }
        div[data-testid="stMetric"] { background: #121e25; border: 1px solid #23353e; padding: 1rem;
                                      border-radius: 12px; }
        div[data-testid="stMetricLabel"] { color: #9aadb5; }
        div[data-testid="stMetricValue"] { color: #f3f7f8; }
        div.stButton > button { border-radius: 8px; font-weight: 700; }
        .note { color: #9aadb5; background: #121e25; border-left: 3px solid #72d7bd;
                padding: .7rem .9rem; border-radius: 6px; font-size: .86rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _initialize_service()
    service: SpindleNadiService = st.session_state.service
    _render_header()
    machine_id = _render_sidebar(service)
    selected = service.latest(machine_id)

    overview_tab, live_tab, what_if_tab, twin_tab, why_tab, copilot_tab, maintenance_tab, timeline_tab, evidence_tab, replay_tab, esp32_tab, alerts_tab = st.tabs(
        [
            "Fleet overview",
            "Live & trends",
            "What-if",
            "Digital twin",
            "Why & RUL",
            "Maintenance copilot",
            "Maintenance history",
            "Evidence timeline",
            "Evidence",
            "Recorded data replay",
            "ESP32 physical data",
            "Alerts & messages",
        ]
    )
    with overview_tab:
        _render_fleet(service, machine_id)
    with live_tab:
        _render_live(service, machine_id)
    with what_if_tab:
        _render_what_if(service, machine_id)
    with twin_tab:
        _render_digital_twin(machine_id, selected)
    with why_tab:
        _render_explanation(service, machine_id, selected)
    with copilot_tab:
        _render_copilot(service, machine_id)
    with timeline_tab:
        _render_timeline(service, machine_id)
    with maintenance_tab:
        _render_maintenance(service, selected)
    with evidence_tab:
        _render_evidence(service, machine_id)
    with replay_tab:
        _render_recorded_replay()
    with esp32_tab:
        _render_esp32_telemetry()
    with alerts_tab:
        _render_alert_history(service, machine_id)


def _initialize_service() -> None:
    if "service" not in st.session_state:
        try:
            try:
                notification_secrets = st.secrets.to_dict()
            except (FileNotFoundError, RuntimeError):
                notification_secrets = {}
            notifier = notifier_from_streamlit_secrets(notification_secrets)
            st.session_state.service = SpindleNadiService(
                pipeline=load_pipeline(),
                notifier=notifier,
            )
            st.session_state.selected_machine = "CNC-01"
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not start Spindle Nadi: {error}")
            st.stop()


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">SPINDLE NADI · MACHINE HEALTH MONITORING</div>
          <div class="hero-title">Feel the Pulse. <span style="color:#72d7bd">Predict the Failure.</span></div>
          <div class="muted">An explainable, simulation-first maintenance decision loop for smaller manufacturing labs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(service: SpindleNadiService) -> str:
    st.sidebar.markdown("## Machine controls")
    machine_id = st.sidebar.selectbox(
        "Selected machine",
        service.machine_ids,
        index=service.machine_ids.index(st.session_state.get("selected_machine", "CNC-01")),
        key="selected_machine",
    )
    simulator = service.simulators[machine_id]
    st.sidebar.caption(f"Scenario: {simulator.scenario.value} · Workload: {simulator.workload:.0f}%")
    st.sidebar.markdown("---")

    if st.sidebar.button("Demo → warning", type="primary", width="stretch"):
        if _perform(lambda: service.advance_until_status(machine_id, "WARNING")):
            st.rerun()
    if st.sidebar.button("Demo → critical", width="stretch"):
        if _perform(lambda: service.advance_until_status(machine_id, "CRITICAL")):
            st.rerun()

    workload = st.sidebar.slider("Workload (%)", 0, 100, int(simulator.workload), key=f"load_{machine_id}")
    if st.sidebar.button("Apply workload", width="stretch"):
        if _perform(lambda: (service.set_workload(machine_id, workload), service.advance(machine_id))):
            st.rerun()

    fault_mode = st.sidebar.selectbox(
        "Fault to simulate",
        FAULT_MODES,
        format_func=lambda mode: FAULT_LABELS[mode],
        key=f"fault_mode_{machine_id}",
    )
    if st.sidebar.button("Start gradual degradation", type="primary", width="stretch"):
        if _perform(lambda: (service.start_degradation(machine_id, fault_mode), service.advance(machine_id))):
            st.rerun()
    if st.sidebar.button("Advance 1 sample", width="stretch"):
        if _perform(lambda: service.advance(machine_id)):
            st.rerun()
    if st.sidebar.button("Run 10 samples", width="stretch"):
        if _perform(lambda: service.advance(machine_id, 10)):
            st.rerun()

    left, right = st.sidebar.columns(2)
    if left.button("Near failure", width="stretch"):
        if _perform(lambda: (service.set_near_failure(machine_id), service.advance(machine_id))):
            st.rerun()
    if right.button("Return normal", width="stretch"):
        if _perform(lambda: service.reset(machine_id)):
            st.rerun()
    if st.sidebar.button("Reset machine", width="stretch"):
        if _perform(lambda: service.reset(machine_id)):
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Prototype state rules")
    st.sidebar.markdown("- NORMAL: risk below 50%")
    st.sidebar.markdown("- WARNING: risk from 50% to below 85%")
    st.sidebar.markdown("- CRITICAL: risk from 85%")
    st.sidebar.caption("50–64% risk is WARNING. These thresholds apply only to this simulated prototype.")
    return machine_id


def _perform(action) -> bool:
    try:
        action()
        return True
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Action failed: {error}")
        return False


def _render_fleet(service: SpindleNadiService, selected_machine: str) -> None:
    records = service.fleet_summary()
    status_counts = {status: sum(record["status"] == status for record in records) for status in STATUS_COLORS}
    cols = st.columns(4)
    for index, (status, count) in enumerate(status_counts.items()):
        cols[index].metric(f"{status.title()} machines", count)
    cols[3].metric("Fleet size", len(records))

    st.subheader("Fleet health")
    st.caption("Machines are prioritized by health state, estimated risk, anomaly score, then shorter estimated RUL.")
    cards = st.columns(2)
    for index, record in enumerate(records):
        color = STATUS_COLORS[record["status"]]
        with cards[index % 2]:
            with st.container(border=True):
                st.caption(f"Priority {index + 1}")
                st.markdown(
                    f'<span class="status-pill" style="color:{color}">{record["status"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {record['machine_id']}")
                st.metric("Failure risk", f"{record['risk']:.0%}")
                st.metric("Prototype health score", f"{record['health_score']}/100")
                st.caption(f"Prototype RUL: {record['rul_hours']:.2f} simulated h")
                st.caption(
                    f"Vibration {record['vibration']:.2f} mm/s · "
                    f"Temperature {record['temperature']:.1f} °C"
                )
                if record["machine_id"] == selected_machine:
                    st.caption("Currently selected")

    st.markdown(
        '<div class="note">Choose a machine in the sidebar, start degradation, then advance samples '
        'to watch its trend and health state change.</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def _render_live(service: SpindleNadiService, machine_id: str) -> None:
    if st.toggle(
        "Auto-stream synthetic readings (one new sample per second)",
        key=f"auto_stream_{machine_id}",
    ):
        try:
            service.advance(machine_id)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Live simulator update failed: {error}")
            return
    record = service.latest(machine_id)
    speech_enabled = st.toggle(
        "Enable spoken Hindi + English status alerts",
        value=False,
        key=f"spoken_alerts_{machine_id}",
        help="Plays a short tone and spoken message only when this machine changes status.",
    )
    status_key = f"spoken_alert_status_{machine_id}"
    previous_status = st.session_state.get(status_key)
    announce_transition = previous_status is not None and previous_status != record["status"]
    st.session_state[status_key] = record["status"]
    if speech_enabled:
        render_status_announcement(
            record["status"],
            machine_id,
            record["risk"],
            str(record["timestamp"]),
            announce=announce_transition,
        )
        st.caption(
            f"Current announcement: {spoken_status_message(record['status'], machine_id, record['risk'])} "
            "· Browser speech availability varies; use the button to replay it."
        )
    color = STATUS_COLORS[record["status"]]
    st.markdown(
        f'<span class="status-pill" style="color:{color}">{record["status"]}</span> '
        f'<span class="muted">· {machine_id} · {record["scenario"]} · '
        f'injected mode: {FAULT_LABELS.get(record["fault_mode"], "None")}</span>',
        unsafe_allow_html=True,
    )
    st.caption("Synthetic simulator data · each streamed sample advances 0.1 simulated operating hours.")
    if record["status"] == "WARNING":
        st.warning(
            f"Dashboard alert · {machine_id} is WARNING at {record['risk']:.0%} estimated risk. "
            "Inspect the latest evidence and plan maintenance."
        )
    elif record["status"] == "CRITICAL":
        st.error(
            f"Critical alert · {machine_id} is CRITICAL at {record['risk']:.0%} estimated risk. "
            "Check Alerts & messages for email/WhatsApp delivery status."
        )
    metrics = [
        ("Temperature", f"{record['temperature']:.1f} °C"),
        ("Vibration", f"{record['vibration']:.2f} mm/s"),
        ("Vibration X", f"{record['vibration_x']:.2f} mm/s"),
        ("Vibration Y", f"{record['vibration_y']:.2f} mm/s"),
        ("Vibration Z", f"{record['vibration_z']:.2f} mm/s"),
        ("Motor current", f"{record['motor_current']:.2f} A"),
        ("RPM", f"{record['rpm']:.0f}"),
        ("Spindle load", f"{record['spindle_load_pct']:.1f}%"),
        ("Torque", f"{record['torque_nm']:.1f} N·m"),
        ("Bearing temperature", f"{record['bearing_temperature_c']:.1f} °C"),
        ("Coolant flow", f"{record['coolant_flow_lpm']:.1f} L/min"),
        ("Coolant temperature", f"{record['coolant_temperature_c']:.1f} °C"),
        ("Lubrication pressure", f"{record['lubrication_pressure_bar']:.2f} bar"),
        ("Acoustic level", f"{record['acoustic_noise_db']:.1f} dB"),
        ("Feed rate", f"{record['feed_rate_mm_min']:.0f} mm/min"),
        ("Estimated tool wear", f"{record['tool_wear_pct_est']:.1f}%"),
        ("Estimated power", f"{record['power_kw_est']:.2f} kW"),
        ("Workload", f"{record['workload']:.0f}%"),
    ]
    for start in range(0, len(metrics), 5):
        columns = st.columns(5)
        for column, (label, value) in zip(columns, metrics[start : start + 5]):
            column.metric(label, value)
    st.caption("Tool wear and power are simulator estimates, not direct sensor measurements.")
    risk_col, health_col, sensor_col = st.columns(3)
    risk_col.metric("Model-estimated failure risk", f"{record['risk']:.0%}")
    health_col.metric("Prototype health score", f"{record['health_score']}/100")
    health_col.caption("100 × (1 − estimated risk); not measured equipment health.")
    quality = service.sensor_health(record)
    sensor_col.metric("Sensor data quality", f"{quality['quality_pct']}%", quality["status"])
    with st.expander("Sensor-quality checks", expanded=quality["status"] != "HEALTHY"):
        st.caption(
            f"{quality['valid_channels']} of {quality['total_channels']} channels pass broad "
            "plausibility checks. These ranges are illustrative, not manufacturer limits."
        )
        if quality["issues"]:
            st.dataframe(pd.DataFrame(quality["issues"]), hide_index=True, width="stretch")
        else:
            st.success("All required channels are present, plausible, and fresh.")
    st.progress(int(record["risk"] * 100), text=f"Risk score · anomaly score {record['anomaly_score']:.0f}/100")
    if record["is_anomaly"]:
        st.warning("Latest sensor pattern is unusual relative to the simulated healthy baseline.")

    history = service.records_for(machine_id)
    if len(history) < 2:
        st.info("Advance more samples to build time-series trends.")
        return
    st.subheader("Estimated risk trend")
    st.line_chart(
        history.set_index("operating_hours")[["risk"]],
        y="risk",
        height=180,
    )
    st.subheader("Sensor trends")
    chart_options = list(SENSOR_LABELS)
    selected_signals = st.multiselect(
        "Trend signals",
        chart_options,
        default=[
            "temperature",
            "vibration",
            "bearing_temperature_c",
            "coolant_flow_lpm",
            "lubrication_pressure_bar",
            "acoustic_noise_db",
        ],
        format_func=lambda column: SENSOR_LABELS[column],
        key=f"trend_signals_{machine_id}",
    )
    if not selected_signals:
        st.info("Choose at least one sensor to draw a trend.")
        return
    chart_data = history.melt(
        id_vars=["operating_hours"],
        value_vars=selected_signals,
        var_name="sensor",
        value_name="reading",
    )
    chart_data["sensor"] = chart_data["sensor"].map(SENSOR_LABELS)
    chart = px.line(
        chart_data,
        x="operating_hours",
        y="reading",
        color="sensor",
        facet_row="sensor",
        height=max(400, 220 * len(selected_signals)),
        color_discrete_sequence=list(CHART_COLORS.values()),
        labels={"operating_hours": "Simulated operating hours", "reading": "Sensor reading"},
    )
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1116",
        plot_bgcolor="#111c22",
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    chart.update_yaxes(matches=None, showgrid=True, gridcolor="#22323a")
    st.plotly_chart(chart, width="stretch")
    st.download_button(
        "Download selected machine CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name=f"{machine_id.lower().replace('-', '_')}_readings.csv",
        mime="text/csv",
    )


@st.cache_data(ttl=300, max_entries=8)
def _load_replay_data(
    contents: bytes,
    filename: str,
    member_name: str | None,
    sheet_name: str | None,
) -> pd.DataFrame:
    return parse_recorded_file(contents, filename, member_name, sheet_name)


def _render_recorded_replay() -> None:
    st.subheader("Replay a benchmark or recorded sensor CSV")
    st.caption(
        "Upload CSV, whitespace-delimited TXT, Excel, or ZIP datasets for visualization. "
        "Real timestamps are preserved; cycle data uses cycle order; files without either use a "
        "synthetic one-second-per-row replay sequence. Uploaded benchmark channels are not scored by "
        "the CNC model unless their meaning and units are independently validated."
    )
    uploaded = st.file_uploader(
        "Recorded or benchmark data file",
        type=["csv", "txt", "xlsx", "zip"],
        key="recorded_csv_upload",
    )
    st.markdown("**Documentation only — PDF files are not parsed as sensor data.**")
    documentation = st.file_uploader(
        "Optional PDF reference / codebook",
        type=["pdf"],
        key="recorded_pdf_upload",
    )
    if documentation is not None:
        st.caption(f"Reference file: {documentation.name} · {len(documentation.getvalue()):,} bytes")
        st.download_button(
            "Download PDF reference",
            data=documentation.getvalue(),
            file_name=Path(documentation.name).name,
            mime="application/pdf",
            key="download_uploaded_pdf_reference",
        )

    if uploaded is None:
        st.info(
            "Use CSV or Excel with headers where possible. Headerless 26-column C-MAPSS TXT files are "
            "recognized as engine ID, cycle, three operating settings, and 21 generically named sensors."
        )
        return

    contents = uploaded.getvalue()
    member_name: str | None = None
    sheet_name: str | None = None
    try:
        if uploaded.name.lower().endswith(".zip"):
            members = list_supported_archive_members(contents)
            if not members:
                st.error("ZIP archive contains no supported CSV, TXT, or XLSX files.")
                return
            member_name = st.selectbox(
                "Choose a data file from the ZIP archive",
                members,
                key=f"replay_member_{uploaded.name}",
            )
            st.caption("Archive data is read in memory; files are not extracted into the project.")
        source_name = member_name or uploaded.name
        if source_name.lower().endswith(".xlsx"):
            workbook_bytes = (
                read_supported_archive_member(contents, member_name)
                if member_name is not None
                else contents
            )
            sheets = list_excel_sheets(workbook_bytes)
    except ValueError as error:
        st.error(f"Data file could not be inspected: {error}")
        return
    else:
        if source_name.lower().endswith(".xlsx"):
            sheet_name = st.selectbox(
                "Excel worksheet",
                sheets,
                key=f"replay_sheet_{uploaded.name}_{member_name or 'direct'}",
            )

    try:
        replay = _load_replay_data(contents, uploaded.name, member_name, sheet_name)
    except ValueError as error:
        st.error(f"Recorded data could not be loaded: {error}")
        return

    available_channels = sensor_columns(replay)
    if not available_channels:
        st.error("No numeric measurement columns were found in this dataset.")
        return
    st.caption(
        f"Loaded {len(replay):,} rows from {source_name}. Columns are standardized only where their "
        "headers have a known meaning; unknown benchmark sensors keep their source names."
    )

    machine_options = list(dict.fromkeys(replay["machine_id"].astype(str)))
    if len(machine_options) > 1:
        selected_machine = st.selectbox("Engine / machine", machine_options, key=f"replay_engine_{uploaded.name}")
        replay_view = replay[replay["machine_id"].astype(str) == selected_machine].reset_index(drop=True)
    else:
        selected_machine = machine_options[0]
        replay_view = replay

    canonical_fields = [
        ("temperature", "Temperature"),
        ("vibration", "Vibration"),
        ("motor_current", "Motor current"),
        ("rpm", "RPM"),
    ]
    with st.expander("Optional column mapping for display"):
        st.caption(
            "Mappings change display labels only. Map an unknown channel to a CNC quantity only if the "
            "dataset codebook confirms its physical meaning and units. No model score is produced from these mappings."
        )
        mappings: dict[str, str] = {}
        duplicate_mapping_sources: set[str] = set()
        for field, label in canonical_fields:
            existing = field if field in available_channels else None
            default_index = available_channels.index(existing) + 1 if existing else 0
            source_column = st.selectbox(
                f"Map source column to {label}",
                ["Not mapped", *available_channels],
                index=default_index,
                key=f"replay_mapping_{uploaded.name}_{field}",
            )
            if source_column != "Not mapped":
                if source_column in mappings:
                    duplicate_mapping_sources.add(source_column)
                else:
                    mappings[source_column] = f"{label} (mapped)"
        if duplicate_mapping_sources:
            st.warning(
                "Each source column can map to only one display label. "
                "Duplicate selections keep the first mapping."
            )

    default_channels = [channel for channel in NUMERIC_CHANNELS if channel in available_channels]
    if not default_channels:
        default_channels = available_channels[: min(5, len(available_channels))]
    selected_channels = st.multiselect(
        "Measurement columns to chart",
        available_channels,
        default=default_channels,
        key=f"replay_channels_{uploaded.name}_{member_name or 'direct'}_{sheet_name or 'default'}",
    )
    if not selected_channels:
        st.info("Select at least one numeric measurement column to chart.")
        return

    time_column = (
        "timestamp" if "timestamp" in replay_view
        else "replay_time" if "replay_time" in replay_view
        else "replay_time_seconds"
    )
    time_label = {
        "timestamp": "Recorded timestamp",
        "replay_time": "Recorded cycle / replay time (source units)",
        "replay_time_seconds": "Synthetic replay time (seconds; 1 second per row)",
    }[time_column]
    if len(replay_view) > 1:
        step = st.select_slider(
            "Replay step size (samples per click)",
            options=[1, 2, 5, 10, 25, 50],
            value=1,
            key=f"replay_speed_{uploaded.name}",
        )
        sample_key = f"recorded_replay_index_{uploaded.name}"
        previous_col, next_col = st.columns(2)
        if previous_col.button("Previous samples", key=f"replay_previous_{uploaded.name}"):
            st.session_state[sample_key] = max(0, st.session_state.get(sample_key, 0) - step)
        if next_col.button("Next samples", key=f"replay_next_{uploaded.name}"):
            st.session_state[sample_key] = min(
                len(replay_view) - 1,
                st.session_state.get(sample_key, 0) + step,
            )
        index = st.slider(
            "Recorded sample",
            min_value=0,
            max_value=len(replay_view) - 1,
            value=0,
            key=sample_key,
        )
    else:
        index = 0
    record = replay_view.iloc[index]
    if time_column == "timestamp":
        time_context = str(record["timestamp"])
    elif time_column == "replay_time":
        time_context = f"cycle/time {record['replay_time']} (source units)"
    else:
        time_context = f"synthetic t={record['replay_time_seconds']:.0f}s"
    st.caption(
        f"Recorded dataset · {record['machine_id']} · {time_context} · "
        f"sample {index + 1} of {len(replay_view)}"
    )
    for start in range(0, len(selected_channels), 4):
        columns = st.columns(4)
        for column, channel in zip(columns, selected_channels[start : start + 4]):
            value = record[channel]
            display = "missing" if pd.isna(value) else f"{value:,.3f}"
            label = mappings.get(channel, SENSOR_LABELS.get(channel, channel.replace("_", " ").title()))
            column.metric(label, display)

    plot_data = replay_view.melt(
        id_vars=[time_column],
        value_vars=selected_channels,
        var_name="sensor",
        value_name="reading",
    )
    plot_data["sensor"] = plot_data["sensor"].map(
        lambda channel: mappings.get(
            channel, SENSOR_LABELS.get(channel, channel.replace("_", " ").title())
        )
    )
    figure = px.line(
        plot_data,
        x=time_column,
        y="reading",
        color="sensor",
        facet_row="sensor",
        height=max(350, 200 * len(selected_channels)),
        labels={
            time_column: time_label,
            "reading": "Recorded value",
        },
    )
    figure.update_layout(template="plotly_dark", showlegend=False)
    figure.update_yaxes(matches=None)
    st.plotly_chart(figure, width="stretch")
    st.dataframe(replay, hide_index=True, width="stretch")
    st.warning(
        "Replay is visualization-only. Unknown sensor names retain their dataset labels; mappings are for "
        "display and do not validate physical meaning, convert units, or enable CNC-model scoring. No "
        "missing values are filled and no physical machine is controlled."
    )


@st.fragment(run_every="500ms")
def _render_esp32_telemetry() -> None:
    st.subheader("ESP32 physical sensor stream")
    st.caption(
        "USB serial JSON Lines receiver. Only values arriving from the selected device are displayed; "
        "these readings are not mixed into synthetic machine data or assigned a model risk score."
    )
    try:
        ports = list_esp32_ports()
    except (ImportError, OSError) as error:
        st.error(f"Could not list serial ports: {error}")
        return

    if not ports:
        st.info("No serial ports detected. Connect the ESP32 by USB and refresh this page.")
    port_labels = {device: f"{device} — {description}" for device, description in ports}
    reader: Esp32SerialReader | None = st.session_state.get("esp32_serial_reader")
    is_connected = reader is not None and not reader.closed
    selected_port = st.selectbox(
        "ESP32 serial port",
        list(port_labels),
        format_func=lambda device: port_labels[device],
        disabled=not ports or is_connected,
        key="esp32_port",
    )
    baudrate = st.selectbox(
        "Baud rate",
        [115200, 9600, 19200, 38400, 57600, 230400],
        disabled=is_connected,
        key="esp32_baudrate",
    )

    start, stop = st.columns(2)
    if start.button("Connect ESP32", type="primary", disabled=is_connected or not ports):
        try:
            st.session_state.esp32_serial_reader = Esp32SerialReader(selected_port, baudrate)
            st.session_state.esp32_serial_records = []
            st.session_state.esp32_serial_errors = []
            st.rerun(scope="fragment")
        except (OSError, ValueError, ImportError) as error:
            st.error(f"ESP32 connection failed: {error}")
    if stop.button("Disconnect ESP32", disabled=not is_connected):
        reader.close()
        st.session_state.esp32_serial_reader = None
        st.rerun(scope="fragment")

    reader = st.session_state.get("esp32_serial_reader")
    if reader is None or reader.closed:
        st.info("Connect a device to begin receiving physical sensor readings.")
        return
    st.success("Connected · waiting for calibrated sensor messages")
    records: list[dict[str, str | float]] = st.session_state.setdefault("esp32_serial_records", [])
    errors: list[str] = st.session_state.setdefault("esp32_serial_errors", [])
    try:
        new_records, new_errors = reader.read_available()
    except (OSError, ValueError) as error:
        st.error(f"ESP32 serial read failed: {error}")
        reader.close()
        st.session_state.esp32_serial_reader = None
        return
    records.extend(new_records)
    errors.extend(new_errors)
    del records[:-1000]
    del errors[:-10]
    if errors:
        st.warning(f"Skipped {len(errors)} invalid serial line(s). Latest issue: {errors[-1]}")
    if not records:
        st.info("No complete sensor messages received yet. Check the ESP32 JSON Lines format and baud rate.")
        return

    latest = records[-1]
    st.caption(
        f"Source: {latest['data_source']} · Device: {latest['machine_id']} · "
        f"Received at: {latest['timestamp']} · {len(records)} samples"
    )
    physical_channels = [channel for channel in PHYSICAL_CHANNELS if channel in latest]
    for offset in range(0, len(physical_channels), 4):
        columns = st.columns(4)
        for column, channel in zip(columns, physical_channels[offset : offset + 4]):
            column.metric(SENSOR_LABELS[channel], f"{latest[channel]:,.3f}")
    if len(records) > 1 and physical_channels:
        frame = pd.DataFrame(records)
        chart_data = frame.melt(
            id_vars=["timestamp"],
            value_vars=physical_channels,
            var_name="sensor",
            value_name="reading",
        )
        chart_data["sensor"] = chart_data["sensor"].map(SENSOR_LABELS)
        chart = px.line(
            chart_data,
            x="timestamp",
            y="reading",
            color="sensor",
            facet_row="sensor",
            height=max(350, 200 * len(physical_channels)),
            labels={"timestamp": "Device timestamp", "reading": "Calibrated sensor value"},
        )
        chart.update_layout(template="plotly_dark", showlegend=False)
        chart.update_yaxes(matches=None)
        st.plotly_chart(chart, width="stretch")
    st.download_button(
        "Download ESP32 readings",
        data=pd.DataFrame(records).to_csv(index=False).encode("utf-8"),
        file_name="spindle_nadi_esp32_physical_readings.csv",
        mime="text/csv",
    )
    st.warning(
        "Send calibrated engineering-unit values with canonical JSON keys. Raw ADC counts must not be "
        "reported as °C, A, or mm/s. A real-machine risk model requires validated training data and "
        "calibrated sensor baselines; this physical stream does not use the synthetic model."
    )


def _render_alert_history(service: SpindleNadiService, machine_id: str) -> None:
    st.subheader("Alert delivery and recovery")
    st.caption(
        "NORMAL produces no message. WARNING stays in the dashboard. CRITICAL and verified post-maintenance "
        "recovery attempt configured email and WhatsApp channels."
    )
    notifier = service.notifier
    if notifier is not None and hasattr(notifier, "configuration_status"):
        configured = notifier.configuration_status()
    else:
        configured = {"email": False, "whatsapp": False}
    email_col, whatsapp_col = st.columns(2)
    email_col.metric("Email", "Configured" if configured["email"] else "Not configured")
    whatsapp_col.metric("WhatsApp", "Configured" if configured["whatsapp"] else "Not configured")
    st.caption(
        "Each channel supports up to 10 recipients. Configure the email `recipients` "
        "and WhatsApp `to_numbers` lists in `.streamlit/secrets.toml`."
    )
    if not any(configured.values()):
        st.info(
            "Add provider settings to `.streamlit/secrets.toml` using the example in "
            "`docs/notifications.md`. Secrets are never shown in this dashboard."
        )

    st.download_button(
        "Download maintenance report",
        data=service.maintenance_report(machine_id).encode("utf-8"),
        file_name=f"{machine_id.lower().replace('-', '_')}_maintenance_report.md",
        mime="text/markdown",
        key=f"maintenance_report_{machine_id}",
    )
    st.caption(
        "The report includes current model evidence, trend, sensor-quality findings, "
        "recommendation, and this machine's alert and maintenance history."
    )

    if st.button("Send test notification", key="send_notification_test"):
        outcome = service.send_test_notification(machine_id)
        details = (
            f"Email: {outcome['email_status']} — {outcome['email_detail']}\n\n"
            f"WhatsApp: {outcome['whatsapp_status']} — {outcome['whatsapp_detail']}"
        )
        if any(
            outcome[key] == "sent"
            for key in ("email_status", "whatsapp_status")
        ):
            st.success(details)
        else:
            st.warning(details)
        st.caption("This sends a real test message on each configured channel and may incur provider charges.")

    history = pd.DataFrame(service.notification_history)
    if history.empty:
        st.info("No WARNING, CRITICAL, recovery, or test alerts have been recorded in this session.")
        return
    display = history.sort_values("timestamp", ascending=False).copy()
    display = display.rename(
        columns={
            "timestamp": "Time",
            "machine_id": "Machine",
            "event_type": "Event",
            "status": "Machine state",
            "risk": "Risk",
            "message": "Message",
            "dashboard_status": "Dashboard",
            "email_status": "Email",
            "email_detail": "Email result",
            "whatsapp_status": "WhatsApp",
            "whatsapp_detail": "WhatsApp result",
        }
    )
    st.dataframe(
        display[
            [
                "Time",
                "Machine",
                "Event",
                "Machine state",
                "Risk",
                "Message",
                "Dashboard",
                "Email",
                "Email result",
                "WhatsApp",
                "WhatsApp result",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download alert history",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="spindle_nadi_notification_history.csv",
        mime="text/csv",
    )


def _render_what_if(service: SpindleNadiService, machine_id: str) -> None:
    st.subheader(f"What if operating conditions change on {machine_id}?")
    current_workload = service.simulators[machine_id].workload
    current_rpm = service.simulators[machine_id].rpm_setpoint
    key = f"what_if_result_{machine_id}"
    with st.form(f"what_if_form_{machine_id}"):
        workload_col, rpm_col = st.columns(2)
        workload = workload_col.slider(
            "Alternative workload (%)", 0, 100, min(100, int(current_workload + 25)),
            key=f"what_if_load_{machine_id}",
        )
        rpm_setpoint = rpm_col.slider(
            "Alternative spindle speed (RPM)", 500, 12_000, int(current_rpm), step=100,
            key=f"what_if_rpm_{machine_id}",
        )
        samples = st.slider(
            "Projection horizon (samples)",
            min_value=1,
            max_value=30,
            value=10,
            key=f"what_if_samples_{machine_id}",
        )
        run_projection = st.form_submit_button("Run paired what-if")
    if run_projection:
        try:
            st.session_state[key] = service.compare_workload(
                machine_id, workload, samples, rpm_setpoint=rpm_setpoint
            )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"What-if projection failed: {error}")
            return

    projection = st.session_state.get(key)
    if not projection:
        st.info("Choose an alternate workload and run a paired projection. Live machine state is not changed.")
        return
    if projection["revision"] != service.revision(machine_id):
        st.info("The machine has changed since this projection. Run the what-if again to compare against current state.")
        return

    baseline = projection["baseline"]
    alternate = projection["what_if"]
    left, right, delta = st.columns(3)
    left.metric("Baseline projected risk", f"{baseline['risk']:.0%}", baseline["status"])
    right.metric("What-if projected risk", f"{alternate['risk']:.0%}", alternate["status"])
    delta.metric("Risk difference", f"{projection['risk_delta']:+.0%}")
    st.caption(
        f"Projected {projection['samples']} paired samples: workload "
        f"{projection['current_workload']:.0f}% → {projection['what_if_workload']:.0f}%; "
        f"spindle setpoint {projection['current_rpm_setpoint']:.0f} → "
        f"{projection['what_if_rpm_setpoint']:.0f} RPM."
    )
    sensor_rows = [
        {
            "Sensor": label,
            "Baseline": round(float(baseline[column]), 2),
            "What-if": round(float(alternate[column]), 2),
            "Change": projection["sensor_deltas"][column],
        }
        for label, column in (
            ("Temperature (°C)", "temperature"),
            ("Vibration (mm/s)", "vibration"),
            ("Motor current (A)", "motor_current"),
            ("RPM", "rpm"),
        )
    ]
    st.dataframe(pd.DataFrame(sensor_rows), hide_index=True, width="stretch")
    st.info(
        "The RPM setpoint changes the projected sensor generation and its operating-point-adjusted "
        "features; workload changes are normalized in the same way. This paired synthetic counterfactual "
        "does not mutate the live machine and is not a physical twin."
    )


def _render_digital_twin(machine_id: str, record: dict) -> None:
    st.subheader(f"Sensor-linked machine view · {machine_id}")
    st.caption(
        "Conceptual digital-twin-style schematic: component markers summarize this sample's model evidence; "
        "this is not a CAD model or physics simulation."
    )
    contribution_by_text = {
        str(item["feature"]).lower().replace("-", " "): float(item["contribution"])
        for item in record["explanations"]
    }
    components = [
        ("Drive motor", 15, 8, ("motor current", "temperature")),
        ("Front bearing", 43, 8, ("vibration",)),
        ("Spindle", 67, 8, ("temperature", "rpm")),
        ("Tool / load", 88, 8, ("vibration", "rpm")),
    ]
    values = []
    for label, x, y, keywords in components:
        contribution = max(
            (
                score
                for feature, score in contribution_by_text.items()
                if any(keyword in feature for keyword in keywords)
            ),
            default=0.0,
        )
        values.append((label, x, y, contribution))

    maximum = max((item[3] for item in values), default=0.0)
    colors = [
        "#38d39f" if maximum == 0 or contribution < maximum * 0.35
        else "#ffc857" if contribution < maximum * 0.7
        else "#ff647c"
        for _, _, _, contribution in values
    ]
    figure = go.Figure()
    figure.add_shape(
        type="rect",
        x0=4,
        y0=4.5,
        x1=96,
        y1=11.5,
        line={"color": "#71848d", "width": 2},
        fillcolor="#16252c",
    )
    for left, right in ((25, 35), (51, 61), (78, 83)):
        figure.add_shape(
            type="rect",
            x0=left,
            y0=5.1,
            x1=right,
            y1=10.9,
            line={"color": "#46606b", "width": 1},
            fillcolor="#21353e",
        )
    figure.add_shape(
        type="line",
        x0=7,
        y0=8,
        x1=94,
        y1=8,
        line={"color": "#9bb1ba", "width": 3},
    )
    figure.add_trace(
        go.Scatter(
            x=[item[1] for item in values],
            y=[item[2] for item in values],
            mode="markers+text",
            text=[item[0] for item in values],
            textposition="top center",
            marker={"size": 30, "color": colors, "line": {"color": "#e9f0f2", "width": 2}},
            customdata=[[item[3], record["risk"], record["status"]] for item in values],
            hovertemplate=(
                "%{text}<br>Risk contribution: %{customdata[0]:.3f}"
                "<br>Overall risk: %{customdata[1]:.0%}<br>Status: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1116",
        plot_bgcolor="#101a20",
        height=370,
        margin={"l": 15, "r": 15, "t": 15, "b": 15},
        xaxis={"visible": False, "range": [0, 100]},
        yaxis={"visible": False, "range": [2, 14]},
    )
    st.plotly_chart(figure, width="stretch")
    values = st.columns(4)
    values[0].metric("Temperature", f"{record['temperature']:.1f} °C")
    values[1].metric("Vibration", f"{record['vibration']:.2f} mm/s")
    values[2].metric("Motor current", f"{record['motor_current']:.2f} A")
    values[3].metric("Spindle speed", f"{record['rpm']:.0f} RPM")


def _render_explanation(service: SpindleNadiService, machine_id: str, record: dict) -> None:
    st.subheader(f"Why is {machine_id} at risk?")
    st.caption("Contributions are estimated from the trained model by restoring one feature at a time to its healthy reference.")
    explanation = record["explanations"]
    contributions = pd.DataFrame(explanation)
    contributions = contributions[contributions["contribution"] > 0]
    if contributions.empty:
        st.success("The model did not identify a positive risk contribution above its healthy reference in this sample.")
    else:
        plot = px.bar(
            contributions.sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            labels={"contribution": "Change in estimated risk", "feature": "Sensor feature"},
            color_discrete_sequence=["#72d7bd"],
        )
        plot.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1116",
            plot_bgcolor="#111c22",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(plot, width="stretch")
        st.dataframe(contributions.sort_values("contribution", ascending=False), hide_index=True, width="stretch")

    st.subheader("Last Good Hour · Prototype RUL")
    left, right, last_good_col = st.columns(3)
    left.metric("Estimated remaining useful life", f"{record['rul_hours']:.2f} simulated hours")
    right.metric("Current model risk", f"{record['risk']:.0%}")
    last_good = service.last_good_hour(machine_id)
    if last_good:
        last_good_col.metric("Last good hour", f"{last_good['operating_hours']:.1f} h")
        st.caption(
            f"First model warning observed at {last_good['first_warning_timestamp']}; "
            f"last below-warning sample was {last_good['timestamp']}."
        )
    else:
        last_good_col.metric("Last good hour", "Not reached")
        st.caption("The current session has not yet crossed the prototype warning boundary.")
    st.warning(
        "This estimate extrapolates the simulator's degradation rate. It is not a failure-time guarantee "
        "and is not validated against real machine history."
    )
    with st.expander("How the prototype estimates RUL"):
        st.write(
            "The simulator adds 0.06 degradation per sample and each sample advances 0.1 simulated hours. "
            "The displayed estimate divides the remaining degradation fraction by that synthetic rate."
        )


def _render_copilot(service: SpindleNadiService, machine_id: str) -> None:
    copilot = service.maintenance_copilot(machine_id)
    st.subheader("Maintenance copilot")
    st.caption("Local, evidence-grounded guidance from the trained prototype model; no cloud service or LLM is used.")
    with st.container(border=True):
        st.markdown(f"**{copilot['status']} · {copilot['risk']:.0%} estimated risk**")
        st.write(copilot["urgency"])
        st.markdown("**Model evidence**")
        for item in copilot["evidence"]:
            st.markdown(f"- {item}")
        st.markdown(f"**Suggested action:** {copilot['recommendation']}")
        st.caption(copilot["disclaimer"])

    chat_key = f"copilot_messages_{machine_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": (
                    f"I'm the local maintenance copilot for {machine_id}. Ask me why risk is changing, "
                    "what to inspect, or what the prototype RUL means."
                ),
            }
        ]
    messages = st.session_state[chat_key]
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask about risk, evidence, RUL, or maintenance",
        key=f"copilot_chat_input_{machine_id}",
        max_chars=500,
    )
    if question:
        messages.append({"role": "user", "content": question})
        try:
            answer = service.answer_copilot_question(machine_id, question)
        except ValueError as error:
            answer = str(error)
        messages.append({"role": "assistant", "content": answer})
        st.rerun()
    if st.button("Clear conversation", key=f"copilot_clear_{machine_id}"):
        st.session_state[chat_key] = []
        st.rerun()

    st.subheader("Virtual maintenance simulator")
    st.caption("Apply a virtual wear reduction, then verify the model-estimated risk and status after the next sample.")
    maintenance_options = [
        "Bearing inspection and service",
        "Cooling and lubrication service",
        "Motor and drive inspection",
        "Full simulated service",
    ]
    selected_service = st.selectbox(
        "Simulated maintenance action",
        maintenance_options,
        key=f"maintenance_action_{machine_id}",
    )
    recovery = st.slider(
        "Simulated wear recovery (%)",
        min_value=50,
        max_value=95,
        value=85,
        key=f"maintenance_recovery_{machine_id}",
    )
    if st.button("Perform virtual maintenance", type="primary", key=f"perform_maintenance_{machine_id}"):
        try:
            outcome = service.perform_virtual_maintenance(
                machine_id,
                selected_service,
                recovery_fraction=recovery / 100,
            )
            st.session_state[f"maintenance_outcome_{machine_id}"] = outcome
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Virtual maintenance failed: {error}")

    outcome = st.session_state.get(f"maintenance_outcome_{machine_id}")
    if outcome:
        if outcome["revision"] != service.revision(machine_id):
            st.caption("Previous recovery verification is historical; machine state has changed since that service.")
            return
        before, after = outcome["before"], outcome["after"]
        columns = st.columns(4)
        columns[0].metric("Risk before", f"{before['risk']:.0%}")
        columns[1].metric("Risk after", f"{after['risk']:.0%}", f"{after['risk'] - before['risk']:+.0%}")
        columns[2].metric("Status before", before["status"])
        columns[3].metric("Status after", after["status"])
        if outcome["recovered"]:
            st.success(
                f"Recovery verified: the model returned to NORMAL after "
                f"{outcome['verification_samples']} post-maintenance samples."
            )
        elif outcome["risk_improved"]:
            st.warning(
                f"Risk improved, but status is still {after['status']}; no RECOVERED notification was sent."
            )
        else:
            st.warning(
                "The model did not return to NORMAL within the verification window; no RECOVERED "
                "notification was sent. Inspect the evidence and do not assume recovery."
            )
        st.caption("Virtual result only. No physical machine action was performed.")


def _render_timeline(service: SpindleNadiService, machine_id: str) -> None:
    st.subheader(f"Evidence timeline · {machine_id}")
    timeline = service.evidence_timeline(machine_id)
    if timeline.empty:
        st.info("No evidence has been recorded for this machine yet.")
        return
    risk_rows = timeline[timeline["event"] == "Sensor reading"].copy()
    if not risk_rows.empty:
        chart = px.line(
            risk_rows,
            x="operating_hours",
            y="risk",
            color="status",
            color_discrete_map=STATUS_COLORS,
            markers=True,
            labels={"operating_hours": "Simulated operating hours", "risk": "Estimated risk"},
        )
        chart.add_hline(
            y=WARNING_THRESHOLD, line_dash="dash", line_color="#ffc857", annotation_text="Warning"
        )
        chart.add_hline(
            y=CRITICAL_THRESHOLD, line_dash="dash", line_color="#ff647c", annotation_text="Critical"
        )
        chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1116",
            plot_bgcolor="#111c22",
            yaxis={"range": [0, 1], "tickformat": ".0%"},
        )
        st.plotly_chart(chart, width="stretch")
    st.dataframe(timeline.iloc[::-1].head(100), hide_index=True, width="stretch")
    lead_time = service.observed_warning_lead_time(machine_id)
    if lead_time is None:
        st.caption("Observed AI-vs-fixed-limit lead time is pending: both a model warning and a fixed-limit crossing are needed.")
    elif lead_time > 0:
        st.metric("Observed model warning lead over fixed limit", f"{lead_time:.2f} simulated hours")
    else:
        st.metric("Observed model warning lead over fixed limit", f"{lead_time:.2f} simulated hours")
    st.caption("Lead time is calculated only from events observed in this machine's recorded session; it is not a prediction.")


def _render_maintenance(service: SpindleNadiService, selected: dict) -> None:
    st.subheader("Maintenance alerts")
    if selected["status"] == "NORMAL":
        st.success(f"{selected['machine_id']} is currently NORMAL. Continue monitoring.")
    else:
        st.warning(
            f"{selected['status']} · {selected['machine_id']} · "
            f"estimated risk {selected['risk']:.0%} — maintenance inspection recommended."
        )
        contributions = selected["explanations"][:2]
        reasons = " + ".join(str(item["feature"]).lower() for item in contributions)
        recommendation = service.recommend_maintenance(contributions)
        st.markdown(f"**Model explanation:** {reasons or 'increasing simulated degradation'}")
        st.markdown(f"**Suggested action:** {recommendation}")

    if service.maintenance_history:
        history = pd.DataFrame(service.maintenance_history)
        st.subheader("Maintenance history")
        st.dataframe(history.tail(100).iloc[::-1], hide_index=True, width="stretch")
        st.download_button(
            "Download maintenance history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="spindle_nadi_maintenance_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No health-state transitions have generated alerts yet.")


def _render_evidence(service: SpindleNadiService, machine_id: str) -> None:
    st.subheader("Measured prototype evaluation")
    metrics = service.pipeline.metrics
    st.caption(
        "Held-out machine evaluation on generated data. These measurements describe this synthetic dataset only."
    )
    cells = st.columns(4)
    cells[0].metric("Risk MAE", f"{metrics['mae']:.3f}")
    cells[1].metric("Precision", f"{metrics['precision']:.2f}")
    cells[2].metric("Recall", f"{metrics['recall']:.2f}")
    cells[3].metric("F1 score", f"{metrics['f1']:.2f}")
    st.caption(
        f"Test set: {int(metrics['test_samples'])} samples from {int(metrics['test_machines'])} held-out machines. "
        f"Failure-class metrics use the {WARNING_THRESHOLD:.0%} WARNING threshold."
    )
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual below warning", "Actual warning or critical"],
        columns=["Predicted below warning", "Predicted warning or critical"],
    )
    st.markdown("**Held-out confusion matrix**")
    st.dataframe(matrix, width="stretch")

    training = service.pipeline.training_readings
    st.subheader("Data understanding")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Training readings", f"{len(training):,}")
    summary_columns[1].metric("Training machines", f"{training['machine_id'].nunique()}")
    summary_columns[2].metric("Missing sensor values", f"{int(training.isna().sum().sum())}")
    summary_columns[3].metric(
        "Workload range",
        f"{training['workload'].min():.0f}–{training['workload'].max():.0f}%",
    )
    sensor_columns = ["temperature", "vibration", "motor_current", "rpm"]
    condition_summary = training.groupby("status", sort=False)[sensor_columns].mean().round(2)
    st.markdown("**Average simulated sensor readings by generated condition**")
    st.dataframe(condition_summary, width="stretch")

    st.subheader("Trend-based model vs fixed-limit comparator")
    current = service.latest(machine_id)
    comparison = service.comparison(current)
    left, right = st.columns(2)
    with left:
        st.markdown("**Spindle Nadi trend model**")
        st.write(f"{current['status']} · risk {current['risk']:.0%}")
        st.caption(f"Anomaly score: {current['anomaly_score']:.0f}/100")
    with right:
        st.markdown("**Illustrative fixed-limit comparator**")
        st.write("ALARM" if comparison["fixed_alarm"] else "No fixed limit crossed")
        st.caption(
            "Crossed: " + (", ".join(comparison["crossed_limits"]) if comparison["crossed_limits"] else "none")
        )
    st.dataframe(
        pd.DataFrame(
            [{"Sensor": "Temperature", "Fixed limit": "75 °C"},
             {"Sensor": "Vibration", "Fixed limit": "3.0 mm/s"},
             {"Sensor": "Motor current", "Fixed limit": "12.5 A"},
             {"Sensor": "RPM", "Fixed limit": "Below 1600"}]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "The fixed limits are illustrative comparison settings, not safety limits. No warning lead time "
        "is claimed; both approaches should be validated on the same real operating data before comparison."
    )

    st.subheader("Export")
    all_records = service.all_session_records()
    if not all_records.empty:
        st.download_button(
            "Download all session readings",
            data=all_records.to_csv(index=False).encode("utf-8"),
            file_name="spindle_nadi_session_readings.csv",
            mime="text/csv",
        )
    st.caption(f"Training data source: {Path(DEFAULT_OUTPUT).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
