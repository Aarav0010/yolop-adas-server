/* ===================================================================
   ADAS Lane Detection Dashboard — Application Logic
   =================================================================== */

(() => {
  'use strict';

  // ── DOM References ──────────────────────────────────────────────
  const dom = {
    // Header
    fpsBadge:        document.getElementById('fpsBadge'),
    fpsValue:        document.getElementById('fpsValue'),
    liveIndicator:   document.getElementById('liveIndicator'),

    // Video
    videoFeed:       document.getElementById('videoFeed'),
    videoOffline:    document.getElementById('videoOffline'),
    overlaySource:   document.getElementById('overlaySource'),
    overlayFrame:    document.getElementById('overlayFrame'),

    // Position
    carIcon:         document.getElementById('carIcon'),
    offsetReadout:   document.getElementById('offsetReadout'),
    laneLeft:        document.getElementById('laneLeft'),
    laneRight:       document.getElementById('laneRight'),

    // Alerts
    alertPanel:      document.getElementById('alertPanel'),
    ldwDot:          document.getElementById('ldwDot'),
    ldwStatus:       document.getElementById('ldwStatus'),
    ldwSeverity:     document.getElementById('ldwSeverity'),
    lcaDot:          document.getElementById('lcaDot'),
    lcaStatus:       document.getElementById('lcaStatus'),
    lcaSeverity:     document.getElementById('lcaSeverity'),
    alertBanner:     document.getElementById('alertBanner'),
    alertBannerIcon: document.getElementById('alertBannerIcon'),
    alertBannerMsg:  document.getElementById('alertBannerMsg'),

    // History
    alertHistory:    document.getElementById('alertHistory'),

    // Source buttons
    btnCamera:       document.getElementById('btnCamera'),
    btnVideo1:       document.getElementById('btnVideo1'),
    btnVideo2:       document.getElementById('btnVideo2'),

    // Footer
    footerLaneWidth:   document.getElementById('footerLaneWidth'),
    footerConfidence:  document.getElementById('footerConfidence'),
    footerProcessing:  document.getElementById('footerProcessing'),
    footerFrame:       document.getElementById('footerFrame'),
  };

  // ── State ───────────────────────────────────────────────────────
  const MAX_HISTORY = 50;
  let alertHistoryEntries = [];
  let lastAlertMessage = '';
  let lastLdwStatus = 'CENTERED';
  let lastLcaStatus = 'CENTERED';
  let eventSource = null;
  let reconnectTimer = null;
  let reconnectDelay = 1000;

  // ── SSE Connection ──────────────────────────────────────────────
  function connectSSE() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`http://${window.location.hostname}:43002/events`);

    eventSource.onopen = () => {
      console.log('[ADAS] SSE connected');
      reconnectDelay = 1000;
      setLiveStatus(true);
    };

    eventSource.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        updateDashboard(data);
      } catch (e) {
        console.warn('[ADAS] Failed to parse SSE data:', e);
      }
    };

    eventSource.onerror = () => {
      console.warn('[ADAS] SSE connection lost, reconnecting…');
      setLiveStatus(false);
      eventSource.close();

      // Exponential backoff reconnect
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(() => {
        connectSSE();
        reconnectDelay = Math.min(reconnectDelay * 1.5, 10000);
      }, reconnectDelay);
    };
  }

  // ── Live indicator ──────────────────────────────────────────────
  function setLiveStatus(connected) {
    dom.liveIndicator.classList.toggle('disconnected', !connected);
    dom.liveIndicator.querySelector('.live-text').textContent = connected ? 'LIVE' : 'OFFLINE';
    // Remove videoOffline toggle here because SSE is not implemented in C++ yet
  }

  // ── Main Update Function ────────────────────────────────────────
  function updateDashboard(d) {
    updateFPS(d.fps);
    updatePosition(d);
    updateAlerts(d);
    updateFooter(d);
    updateOverlay(d);
  }

  // ── FPS ─────────────────────────────────────────────────────────
  function updateFPS(fps) {
    if (fps == null) return;
    const val = parseFloat(fps).toFixed(1);
    dom.fpsValue.textContent = val;

    // Color coding
    dom.fpsBadge.classList.remove('fps-green', 'fps-yellow', 'fps-red');
    if (fps > 24)       dom.fpsBadge.classList.add('fps-green');
    else if (fps > 15)  dom.fpsBadge.classList.add('fps-yellow');
    else                dom.fpsBadge.classList.add('fps-red');

    // Subtle pulse
    dom.fpsBadge.classList.remove('pulse');
    void dom.fpsBadge.offsetWidth; // reflow to re-trigger
    dom.fpsBadge.classList.add('pulse');
  }

  // ── Vehicle Position ────────────────────────────────────────────
  function updatePosition(d) {
    // Shift car based on lateral_offset_ratio (-1 … +1 range, clamped)
    const ratio = clamp(d.lateral_offset_ratio || 0, -1, 1);
    // Map ratio to pixel shift: ratio * half of viz width (approx ±30% of container)
    const shiftPercent = ratio * 28; // percentage shift
    dom.carIcon.style.transform = `translateX(calc(-50% + ${shiftPercent}%))`;

    // Offset readout
    const offset = d.lateral_offset_px != null ? d.lateral_offset_px.toFixed(1) : '0.0';
    dom.offsetReadout.textContent = `${offset} px`;

    // Lane detection indicators
    dom.laneLeft.classList.toggle('not-detected', !d.left_lane_detected);
    dom.laneRight.classList.toggle('not-detected', !d.right_lane_detected);
  }

  // ── Alerts ──────────────────────────────────────────────────────
  function updateAlerts(d) {
    // LDW
    const ldwSev = (d.ldw_severity || 'SAFE').toUpperCase();
    dom.ldwStatus.textContent = formatStatus(d.ldw_status);
    dom.ldwSeverity.textContent = ldwSev;
    dom.ldwSeverity.className = 'alert-severity ' + severityClass(ldwSev);
    dom.ldwDot.className = 'status-dot ' + dotClass(ldwSev);

    // LCA
    const lcaSev = (d.lca_severity || 'NORMAL').toUpperCase();
    dom.lcaStatus.textContent = formatStatus(d.lca_status);
    dom.lcaSeverity.textContent = lcaSev;
    dom.lcaSeverity.className = 'alert-severity ' + severityClass(lcaSev);
    dom.lcaDot.className = 'status-dot ' + dotClass(lcaSev);

    // Panel glow based on worst severity
    dom.alertPanel.classList.remove('glow-warning', 'glow-danger');
    if (ldwSev === 'DANGER' || lcaSev === 'CRITICAL') {
      dom.alertPanel.classList.add('glow-danger');
    } else if (ldwSev === 'WARNING' || lcaSev === 'WARNING') {
      dom.alertPanel.classList.add('glow-warning');
    }

    // Alert banner
    const msg = d.alert_message || '';
    if (msg) {
      dom.alertBanner.classList.remove('hidden');
      dom.alertBannerMsg.textContent = msg;
      dom.alertBannerIcon.textContent = ldwSev === 'DANGER' ? '🚨' : '⚠️';
    } else {
      dom.alertBanner.classList.add('hidden');
    }

    // History tracking — log changes in status
    if (d.ldw_status && d.ldw_status !== lastLdwStatus && d.ldw_status !== 'CENTERED') {
      addHistory(d.ldw_severity || 'WARNING', `LDW: ${formatStatus(d.ldw_status)}`);
    }
    if (d.lca_status && d.lca_status !== lastLcaStatus && d.lca_status !== 'CENTERED') {
      addHistory(d.lca_severity || 'NOTICE', `LCA: ${formatStatus(d.lca_status)}`);
    }
    if (msg && msg !== lastAlertMessage) {
      addHistory(ldwSev === 'DANGER' ? 'DANGER' : 'WARNING', msg);
    }
    lastLdwStatus = d.ldw_status || 'CENTERED';
    lastLcaStatus = d.lca_status || 'CENTERED';
    lastAlertMessage = msg;
  }

  function addHistory(severity, message) {
    // Deduplicate recent
    if (alertHistoryEntries.length > 0 && alertHistoryEntries[0].message === message) return;

    const entry = {
      time: new Date(),
      severity,
      message,
    };
    alertHistoryEntries.unshift(entry);
    if (alertHistoryEntries.length > MAX_HISTORY) alertHistoryEntries.pop();
    renderHistory();
  }

  function renderHistory() {
    if (alertHistoryEntries.length === 0) {
      dom.alertHistory.innerHTML = '<div class="history-empty">No alerts recorded yet.</div>';
      return;
    }

    dom.alertHistory.innerHTML = alertHistoryEntries.map((e, i) => {
      const timeStr = e.time.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const dotColor = severityToColor(e.severity);
      return `
        <div class="history-entry" style="animation-delay: ${i * 30}ms">
          <span class="history-dot" style="background:${dotColor}"></span>
          <span class="history-time">${timeStr}</span>
          <span class="history-msg">${escapeHtml(e.message)}</span>
        </div>`;
    }).join('');
  }

  // ── Footer ──────────────────────────────────────────────────────
  function updateFooter(d) {
    dom.footerLaneWidth.textContent   = d.lane_width_px != null ? `${d.lane_width_px} px` : '-- px';
    dom.footerConfidence.textContent  = d.lane_confidence != null ? `${(d.lane_confidence * 100).toFixed(0)}%` : '--%';
    dom.footerProcessing.textContent  = d.processing_ms != null ? `${d.processing_ms.toFixed(1)} ms` : '-- ms';
    dom.footerFrame.textContent       = `${d.frame_number ?? '--'} / ${d.total_frames ?? '--'}`;
  }

  // ── Video Overlay ───────────────────────────────────────────────
  function updateOverlay(d) {
    dom.overlaySource.textContent = d.source_name || '--';
    dom.overlayFrame.textContent  = `Frame ${d.frame_number ?? '--'} / ${d.total_frames ?? '--'}`;
  }

  // ── Source Switching ────────────────────────────────────────────
  function switchSource(source) {
    const buttons = [dom.btnCamera, dom.btnVideo1, dom.btnVideo2];
    buttons.forEach(btn => {
      btn.classList.remove('active');
      btn.classList.add('loading');
    });

    let typeParam = source.startsWith('video') ? 'video' : 'camera';
    fetch(`/switch_source?type=${typeParam}`, {
      method: 'POST'
    })
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(() => {
      // Set active button
      buttons.forEach(btn => {
        btn.classList.remove('loading');
        btn.classList.toggle('active', btn.dataset.source === source);
      });

      // Reload video feed to pick up new source
      const feed = dom.videoFeed;
      const src = feed.src;
      feed.src = '';
      setTimeout(() => { 
        feed.src = src; 
        dom.videoOffline.classList.add('hidden'); // Ensure it hides after switching
      }, 100);
    })
    .catch(err => {
      console.error('[ADAS] Source switch failed:', err);
      buttons.forEach(btn => btn.classList.remove('loading'));
      // Restore last active
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────
  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  function formatStatus(status) {
    if (!status) return '--';
    return status.replace(/_/g, ' ');
  }

  function severityClass(sev) {
    switch (sev) {
      case 'SAFE':     case 'NORMAL':  return 'severity-safe';
      case 'NOTICE':                   return 'severity-notice';
      case 'WARNING':                  return 'severity-warning';
      case 'DANGER':   case 'CRITICAL':return 'severity-danger';
      default:                         return 'severity-safe';
    }
  }

  function dotClass(sev) {
    switch (sev) {
      case 'SAFE':    case 'NORMAL':  return 'safe';
      case 'NOTICE':                  return 'safe';
      case 'WARNING':                 return 'warning';
      case 'DANGER':  case 'CRITICAL':return 'danger';
      default:                        return 'safe';
    }
  }

  function severityToColor(sev) {
    switch ((sev || '').toUpperCase()) {
      case 'SAFE':    case 'NORMAL': return '#00d68f';
      case 'NOTICE':                 return '#00b4d8';
      case 'WARNING':                return '#ffaa00';
      case 'DANGER':  case 'CRITICAL': return '#ff3d71';
      default:                       return '#94a3b8';
    }
  }

  function escapeHtml(str) {
    const el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
  }

  // ── Video Error Handling ────────────────────────────────────────
  // Removed error listener because MJPEG does not fire 'load' to reset the error state reliably.


  // ── Event Listeners ─────────────────────────────────────────────
  dom.btnCamera.addEventListener('click', () => switchSource('camera'));
  dom.btnVideo1.addEventListener('click', () => switchSource('video1'));
  dom.btnVideo2.addEventListener('click', () => switchSource('video2'));

  // ── Initialise ──────────────────────────────────────────────────
  connectSSE();

})();
