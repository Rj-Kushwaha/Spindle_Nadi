"""Parse benchmark and recorded data for safe, read-only replay."""

from __future__ import annotations

from io import BytesIO, StringIO
import re
import zipfile

import numpy as np
import pandas as pd

ALIASES = {
    "timestamp": {
        "timestamp", "time", "datetime", "date", "date_time", "recorded_at",
        "event_time", "sample_time", "created_at",
    },
    "machine_id": {"machine_id", "machine", "asset_id", "engine_id", "unit_number"},
    "cycle": {"cycle", "cycles", "cycle_number", "replay_time", "time_step"},
    "temperature": {"temperature", "temperature_c", "temp_c"},
    "vibration": {"vibration", "vibration_mm_s", "overall_vibration"},
    "vibration_x": {"vibration_x", "vibration_x_mm_s"},
    "vibration_y": {"vibration_y", "vibration_y_mm_s"},
    "vibration_z": {"vibration_z", "vibration_z_mm_s"},
    "motor_current": {"motor_current", "current", "current_a"},
    "rpm": {"rpm", "spindle_speed", "spindle_speed_rpm"},
    "spindle_load_pct": {"spindle_load_pct", "spindle_load", "load_pct"},
    "torque_nm": {"torque_nm", "torque"},
    "bearing_temperature_c": {"bearing_temperature_c", "bearing_temp_c"},
    "coolant_flow_lpm": {"coolant_flow_lpm", "coolant_flow"},
    "coolant_temperature_c": {"coolant_temperature_c", "coolant_temp_c"},
    "lubrication_pressure_bar": {"lubrication_pressure_bar", "lubrication_pressure"},
    "acoustic_noise_db": {"acoustic_noise_db", "acoustic_noise", "noise_db"},
    "feed_rate_mm_min": {"feed_rate_mm_min", "feed_rate"},
    "tool_wear_pct_est": {"tool_wear_pct_est", "tool_wear_pct"},
    "power_kw_est": {"power_kw_est", "power_kw"},
    "workload": {"workload", "workload_pct"},
    "operating_hours": {"operating_hours", "run_hours"},
}
NUMERIC_CHANNELS = tuple(
    name for name in ALIASES if name not in {"timestamp", "machine_id"}
)
REPLAY_METADATA_COLUMNS = {
    "timestamp", "machine_id", "sample_index", "replay_time",
    "replay_time_seconds", "time_source",
}
SUPPORTED_DATA_EXTENSIONS = {".csv", ".txt", ".xlsx"}
MAX_DATA_FILE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1000

_CMAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    *(f"sensor_{index}" for index in range(1, 22)),
]


def list_supported_archive_members(contents: bytes) -> list[str]:
    """List supported ZIP members without extracting them to disk."""
    if not contents:
        raise ValueError("The uploaded ZIP archive is empty")
    try:
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            members = [
                item.filename
                for item in archive.infolist()
                if not item.is_dir()
                and not item.filename.startswith("__MACOSX/")
                and _extension(item.filename) in SUPPORTED_DATA_EXTENSIONS
            ]
            if len(archive.infolist()) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("ZIP archive contains too many entries (maximum 1000)")
            for item in archive.infolist():
                if item.filename in members and item.file_size > MAX_DATA_FILE_BYTES:
                    raise ValueError(f"ZIP member {item.filename!r} exceeds the 100 MB size limit")
            return members
    except zipfile.BadZipFile as error:
        raise ValueError("The uploaded ZIP file is invalid or damaged") from error


def read_supported_archive_member(contents: bytes, member_name: str) -> bytes:
    """Read one supported ZIP member in memory without extracting filesystem paths."""
    try:
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            info = archive.getinfo(member_name)
            if info.is_dir() or _extension(member_name) not in SUPPORTED_DATA_EXTENSIONS:
                raise ValueError("The selected ZIP member is not a supported data file")
            if info.file_size > MAX_DATA_FILE_BYTES:
                raise ValueError("Selected ZIP member exceeds the 100 MB size limit")
            return archive.read(info)
    except zipfile.BadZipFile as error:
        raise ValueError("The uploaded ZIP file is invalid or damaged") from error
    except KeyError as error:
        raise ValueError("Selected data file was not found in the ZIP archive") from error


def list_excel_sheets(contents: bytes) -> list[str]:
    """Return worksheet names from an uploaded Excel workbook."""
    try:
        with pd.ExcelFile(BytesIO(contents), engine="openpyxl") as workbook:
            return workbook.sheet_names
    except (ImportError, ValueError, OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Could not read Excel workbook: {error}") from error


def parse_recorded_file(
    contents: bytes,
    filename: str,
    member_name: str | None = None,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Load CSV, text, or Excel data (including a selected ZIP member)."""
    if not contents:
        raise ValueError("The uploaded file is empty")
    if len(contents) > MAX_DATA_FILE_BYTES:
        raise ValueError("Data file exceeds the 100 MB size limit")

    name = filename
    payload = contents
    extension = _extension(filename)
    if extension == ".zip":
        if not member_name:
            raise ValueError("Choose a supported data file inside the ZIP archive")
        payload = read_supported_archive_member(contents, member_name)
        name = member_name
        extension = _extension(member_name)

    if extension == ".csv":
        raw = _read_delimited(payload, ",", filename=name)
    elif extension == ".txt":
        raw = _read_text_table(payload, name)
    elif extension == ".xlsx":
        try:
            raw = pd.read_excel(BytesIO(payload), sheet_name=sheet_name or 0, engine="openpyxl")
        except (ImportError, ValueError, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"Could not read Excel worksheet: {error}") from error
    else:
        raise ValueError("Supported machine-data files are CSV, TXT, XLSX, or ZIP")

    return standardize_recorded_frame(raw)


def parse_recorded_csv(contents: bytes) -> pd.DataFrame:
    """Compatibility wrapper for CSV uploads."""
    return parse_recorded_file(contents, "recorded.csv")


def standardize_recorded_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize known metadata and sensor aliases while preserving raw columns."""
    if raw.empty:
        raise ValueError("The uploaded file has no data rows")

    normalized_headers: dict[str, str] = {}
    for column in raw.columns:
        normalized = _normalize_header(column)
        for canonical, names in ALIASES.items():
            if normalized in names:
                if canonical in normalized_headers:
                    raise ValueError(f"Multiple columns map to {canonical}")
                normalized_headers[canonical] = column
                break

    result = pd.DataFrame(index=raw.index)
    machine_column = normalized_headers.get("machine_id")
    if machine_column is not None:
        result["machine_id"] = raw[machine_column].fillna("Recorded dataset").astype(str)
    else:
        result["machine_id"] = "Recorded dataset"

    timestamp_column = normalized_headers.get("timestamp")
    cycle_column = normalized_headers.get("cycle")
    timestamp_values: pd.Series | None = None
    if timestamp_column is not None:
        source_times = raw[timestamp_column]
        numeric_times = pd.to_numeric(source_times, errors="coerce")
        if numeric_times.notna().all():
            if cycle_column is None:
                cycle_column = timestamp_column
        else:
            timestamp_values = pd.to_datetime(source_times, errors="coerce", utc=True)
            if timestamp_values.isna().any():
                raise ValueError("Timestamp column contains invalid date/time values")

    if timestamp_values is not None:
        result["timestamp"] = timestamp_values

    if cycle_column is not None:
        cycle_values = pd.to_numeric(raw[cycle_column], errors="coerce")
        if cycle_values.notna().any():
            result["cycle"] = cycle_values

    recognized_source_columns = set(normalized_headers.values())
    for column in raw.columns:
        if column in recognized_source_columns:
            canonical = next(
                (name for name, source in normalized_headers.items() if source == column),
                None,
            )
            if canonical not in {"timestamp", "machine_id", "cycle"}:
                values = pd.to_numeric(raw[column], errors="coerce")
                if values.notna().any():
                    result[canonical] = values
            continue

        series = raw[column]
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().any():
            output_name = _unique_column_name(str(column), result.columns)
            result[output_name] = values
        elif str(column) not in result.columns:
            result[str(column)] = series

    numeric_columns = sensor_columns(result)
    if not numeric_columns:
        raise ValueError(
            "File must contain at least one recognized numeric sensor or benchmark measurement column"
        )

    result = result.reset_index(drop=True)
    result["sample_index"] = np.arange(1, len(result) + 1)
    if "timestamp" in result:
        return result.sort_values("timestamp", kind="stable").reset_index(drop=True)

    if "cycle" in result:
        result["replay_time"] = result["cycle"]
        result["time_source"] = "cycle"
        return result.sort_values(["machine_id", "replay_time"], kind="stable").reset_index(drop=True)

    result["replay_time_seconds"] = np.arange(len(result), dtype=float)
    result["time_source"] = "synthetic sequence (1 second/sample)"
    return result


def sensor_columns(frame: pd.DataFrame) -> list[str]:
    """Return finite numeric candidate columns without interpreting unknown sensor semantics."""
    return [
        column
        for column in frame.columns
        if column not in REPLAY_METADATA_COLUMNS
        and column != "cycle"
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]


def _read_delimited(contents: bytes, delimiter: str, filename: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(BytesIO(contents), sep=delimiter)
        if _first_line_is_numeric(contents, delimiter):
            raw = pd.read_csv(BytesIO(contents), sep=delimiter, header=None)
            raw.columns = (
                _CMAPSS_COLUMNS
                if raw.shape[1] == len(_CMAPSS_COLUMNS) and _looks_like_cmapss_filename(filename)
                else [f"measurement_{index + 1}" for index in range(raw.shape[1])]
            )
        return raw
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not parse {filename}: {error}") from error


def _read_text_table(contents: bytes, filename: str) -> pd.DataFrame:
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Could not decode {filename} as UTF-8 text") from error
    if not text.strip():
        raise ValueError(f"{filename} contains no data")

    first_line = next(line for line in text.splitlines() if line.strip())
    fields = re.split(r"\s+", first_line.strip())
    if len(fields) == 26 and all(_is_number(value) for value in fields):
        raw = pd.read_csv(StringIO(text), sep=r"\s+", header=None)
        if raw.shape[1] != len(_CMAPSS_COLUMNS):
            raise ValueError("C-MAPSS text rows must contain 26 numeric values")
        raw.columns = _CMAPSS_COLUMNS
        return raw
    try:
        return pd.read_csv(StringIO(text), sep=r"\s+")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not parse whitespace-delimited text {filename}: {error}") from error


def _first_line_is_numeric(contents: bytes, delimiter: str) -> bool:
    try:
        first_line = contents.decode("utf-8-sig").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return False
    values = first_line.split(delimiter)
    return len(values) > 1 and all(_is_number(value.strip()) for value in values)


def _normalize_header(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _looks_like_cmapss_filename(filename: str) -> bool:
    normalized = _normalize_header(filename)
    return "cmapss" in normalized or any(f"fd00{index}" in normalized for index in range(1, 5))


def _unique_column_name(name: str, existing: pd.Index) -> str:
    if name not in existing:
        return name
    suffix = 2
    while f"{name}_{suffix}" in existing:
        suffix += 1
    return f"{name}_{suffix}"


def _extension(filename: str) -> str:
    return f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""


def _is_number(value: str) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
