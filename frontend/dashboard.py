"""Fleet monitoring dashboard for the Spindle Nadi prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.service import SpindleNadiService
from data.generate_data import DEFAULT_OUTPUT, generate_dataset
from ml.pipeline import SpindleNadiPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_COLORS = {"NORMAL": "#38d39f", "WARNING": "#ffc857", "CRITICAL": "#ff647c"}
CHART_COLORS = {
    "temperature": "#ff8a65",
    "vibration": "#b69cff",
    "motor_current": "#4dc6d5",
    "rpm": "#62a5ff",
}


@st.cache_resource
def load_pipeline() -> SpindleNadiPipeline:
    training_path = Path(DEFAULT_OUTPUT)
    if not training_path.exists():
        generate_dataset(training_path)
    return SpindleNadiPipeline(pd.read_csv(training_path))


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

    overview_tab, live_tab, what_if_tab, twin_tab, why_tab, copilot_tab, maintenance_tab, timeline_tab, evidence_tab = st.tabs(
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
        ]
    )
    with overview_tab:
        _render_fleet(service, machine_id)
    with live_tab:
        _render_live(service, machine_id, selected)
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


def _initialize_service() -> None:
    if "service" not in st.session_state:
        try:
            st.session_state.service = SpindleNadiService(pipeline=load_pipeline())
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

    if st.sidebar.button("Start gradual degradation", type="primary", width="stretch"):
        if _perform(lambda: (service.start_degradation(machine_id), service.advance(machine_id))):
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
    st.sidebar.markdown("- NORMAL: risk below 40%")
    st.sidebar.markdown("- WARNING: risk from 40%")
    st.sidebar.markdown("- CRITICAL: risk from 80%")
    st.sidebar.caption("These thresholds apply only to this simulated prototype.")
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


def _render_live(service: SpindleNadiService, machine_id: str, record: dict) -> None:
    color = STATUS_COLORS[record["status"]]
    st.markdown(
        f'<span class="status-pill" style="color:{color}">{record["status"]}</span> '
        f'<span class="muted">· {machine_id} · {record["scenario"]}</span>',
        unsafe_allow_html=True,
    )
    columns = st.columns(5)
    columns[0].metric("Temperature", f"{record['temperature']:.1f} °C")
    columns[1].metric("Vibration", f"{record['vibration']:.2f} mm/s")
    columns[2].metric("Motor current", f"{record['motor_current']:.2f} A")
    columns[3].metric("RPM", f"{record['rpm']:.0f}")
    columns[4].metric("Workload", f"{record['workload']:.0f}%")
    st.metric("Model-estimated failure risk", f"{record['risk']:.0%}")
    st.progress(int(record["risk"] * 100), text=f"Risk score · anomaly score {record['anomaly_score']:.0f}/100")
    if record["is_anomaly"]:
        st.warning("Latest sensor pattern is unusual relative to the simulated healthy baseline.")

    history = service.records_for(machine_id)
    if len(history) < 2:
        st.info("Advance more samples to build time-series trends.")
        return
    st.subheader("Sensor trends")
    chart_data = history.melt(
        id_vars=["timestamp", "operating_hours"],
        value_vars=["temperature", "vibration", "motor_current", "rpm"],
        var_name="sensor",
        value_name="reading",
    )
    chart_data["sensor"] = chart_data["sensor"].replace(
        {"temperature": "Temperature °C", "vibration": "Vibration mm/s", "motor_current": "Current A", "rpm": "RPM"}
    )
    chart = px.line(
        chart_data,
        x="operating_hours",
        y="reading",
        color="sensor",
        facet_row="sensor",
        height=680,
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


def _render_what_if(service: SpindleNadiService, machine_id: str) -> None:
    st.subheader(f"What if workload changes on {machine_id}?")
    current_workload = service.simulators[machine_id].workload
    key = f"what_if_result_{machine_id}"
    with st.form(f"what_if_form_{machine_id}"):
        workload = st.slider(
            "Alternative workload (%)",
            min_value=0,
            max_value=100,
            value=min(100, int(current_workload + 25)),
            key=f"what_if_load_{machine_id}",
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
            st.session_state[key] = service.compare_workload(machine_id, workload, samples)
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
        f"Projected {projection['samples']} samples at {projection['current_workload']:.0f}% "
        f"baseline workload vs {projection['what_if_workload']:.0f}% alternative workload."
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
        "Workload-adjusted features are designed to distinguish expected load-related sensor changes "
        "from degradation. A workload increase can raise raw temperature/current without necessarily "
        "raising the model's estimated risk. This is a synthetic counterfactual, not a physical twin."
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

    question = st.text_input(
        "Ask about this machine",
        placeholder="Why is risk increasing? What maintenance is suggested?",
        key=f"copilot_question_{machine_id}",
    )
    if st.button("Ask copilot", key=f"copilot_ask_{machine_id}"):
        try:
            st.session_state[f"copilot_answer_{machine_id}"] = service.answer_copilot_question(machine_id, question)
        except ValueError as error:
            st.error(str(error))
    answer = st.session_state.get(f"copilot_answer_{machine_id}")
    if answer:
        st.info(answer)

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
            st.success("Recovery verified: the simulated model risk decreased after virtual maintenance.")
        else:
            st.warning("No risk reduction was measured after service; inspect the evidence and do not assume recovery.")
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
        chart.add_hline(y=0.4, line_dash="dash", line_color="#ffc857", annotation_text="Warning")
        chart.add_hline(y=0.8, line_dash="dash", line_color="#ff647c", annotation_text="Critical")
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
        "Failure-class metrics use the 40% WARNING threshold."
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
