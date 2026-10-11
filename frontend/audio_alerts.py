"""Browser speech and tone announcements for machine status transitions."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.alert_messages import _SPOKEN_MESSAGES

_ALERT_PLAYER = st.components.v2.component(
    "spindle_nadi_status_announcement",
    html='<div><button id="play-alert" type="button">Play announcement</button></div>',
    css="""
    button {
      background: #17252c;
      border: 1px solid #72d7bd;
      border-radius: 8px;
      color: #e8eff2;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 0.55rem 0.85rem;
    }
    button:focus-visible {
      outline: 2px solid #72d7bd;
      outline-offset: 2px;
    }
    """,
    js="""
    const lastEvents = new WeakMap()

    export default function (component) {
      const { data, parentElement } = component
      const button = parentElement.querySelector("#play-alert")
      if (!button) return

      const messages = Array.isArray(data?.messages) ? data.messages : []
      const status = String(data?.status ?? "NORMAL")
      button.textContent = `Play announcement · ${status}`

      function playTone() {
        try {
          const AudioContextClass = window.AudioContext || window.webkitAudioContext
          if (!AudioContextClass) return false
          const context = new AudioContextClass()
          const patterns = {
            NORMAL: [660],
            WARNING: [520, 520],
            CRITICAL: [390, 390, 390],
          }
          const frequencies = patterns[status] || patterns.NORMAL
          frequencies.forEach((frequency, index) => {
            const start = context.currentTime + index * 0.22
            const oscillator = context.createOscillator()
            const gain = context.createGain()
            oscillator.type = "sine"
            oscillator.frequency.value = frequency
            gain.gain.setValueAtTime(0.0001, start)
            gain.gain.exponentialRampToValueAtTime(0.12, start + 0.015)
            gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.14)
            oscillator.connect(gain)
            gain.connect(context.destination)
            oscillator.start(start)
            oscillator.stop(start + 0.15)
          })
          window.setTimeout(() => context.close(), frequencies.length * 240 + 300)
          return true
        } catch {
          return false
        }
      }

      function announce() {
        const tonePlayed = playTone()
        if (!("speechSynthesis" in window)) {
          button.textContent = `${tonePlayed ? "Tone played" : "Audio unavailable"} · speech unavailable · ${status}`
          return
        }
        window.speechSynthesis.cancel()
        messages.forEach((message) => {
          const utterance = new SpeechSynthesisUtterance(String(message.text))
          utterance.lang = String(message.lang)
          window.speechSynthesis.speak(utterance)
        })
      }

      button.onclick = announce
      const eventId = String(data?.event_id ?? "")
      if (data?.announce && eventId && lastEvents.get(parentElement) !== eventId) {
        lastEvents.set(parentElement, eventId)
        announce()
      }
    }
    """,
)

def render_status_announcement(
    status: str,
    machine_id: str,
    risk: float,
    event_id: str,
    *,
    announce: bool,
) -> Any:
    """Render browser speech/tone controls for one machine-state event."""
    if status not in _SPOKEN_MESSAGES:
        raise ValueError(f"Unsupported machine status: {status}")
    return _ALERT_PLAYER(
        data={
            "status": status,
            "event_id": event_id,
            "announce": announce,
            "messages": list(_SPOKEN_MESSAGES[status]),
        },
        key=f"status-announcement-{machine_id}",
    )
