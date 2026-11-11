/* Memory-bounded gravity viewer.
 *
 * The manifest and viewer_summary.json are intentionally the only large-scale
 * metadata read during startup. CSV partitions are parsed in a Web Worker and
 * cached per bundle identity/time slot in IndexedDB. The viewer never builds a
 * JavaScript array for the complete OD table.
 */
(() => {
  'use strict';

  const CACHE_VERSION = 2;
  const REQUIRED = ['bundle_manifest.json', 'model_specification.json', 'report.json'];
  const $ = (id) => document.getElementById(id);
  const status = $('status');
  const leafletScript = document.createElement('script');
  leafletScript.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  leafletScript.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
  leafletScript.crossOrigin = '';
  document.head.appendChild(leafletScript);
  const workerSource = `
    const parseLine = (source) => {
      const cells = []; let cell = ''; let quoted = false;
      for (let i = 0; i < source.length; i += 1) {
        const char = source[i];
        if (char === '"') {
          if (quoted && source[i + 1] === '"') { cell += '"'; i += 1; }
          else quoted = !quoted;
        } else if (char === ',' && !quoted) { cells.push(cell); cell = ''; }
        else cell += char;
      }
      cells.push(cell); return cells;
    };
    self.onmessage = async (event) => {
      const {file, filters, limit, chunkSize, offset = 0, scanAll = false} = event.data;
      let header = null; let carry = ''; let matched = 0; let sent = 0;
      const accept = (row) => Object.entries(filters || {}).every(([key, value]) => String(row[key] ?? '') === String(value));
      const emit = (lines, final = false) => {
        const rows = [];
        for (const line of lines) {
          if (!line) continue;
          const cells = parseLine(line);
          if (!header) { header = cells; continue; }
          const row = Object.fromEntries(header.map((name, index) => [name, cells[index] ?? '']));
          if (accept(row)) { if (matched >= offset && sent < limit) { rows.push(row); sent += 1; } matched += 1; }
        }
        if (rows.length) self.postMessage({type: 'rows', rows});
        if (final) self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
      };
      for (let offset = 0; offset < file.size && (scanAll || sent < limit); offset += chunkSize) {
        carry += await file.slice(offset, Math.min(file.size, offset + chunkSize)).text();
        const lines = carry.split(/\\r?\\n/); carry = lines.pop() || ''; emit(lines);
      }
      if (carry) emit([carry], true); else self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
    };
  `;
  const workerUrl = URL.createObjectURL(new Blob([workerSource], {type: 'text/javascript'}));

  const viewer = {
    files: [], fileMap: new Map(), manifest: null, summary: {}, model: {}, report: {},
    map: null, stops: [], stopsById: new Map(), markerLayer: null, selectedOd: null,
    cacheDb: null, cacheIdentity: '', odRows: [], routeRows: [], sectionPage: 0, sectionPageSize: 500,
    sectionFeatures: [], sectionLayer: null, selectedSection: null,
  };

  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const num = (value) => { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; };
  const display = (value) => { const parsed = num(value); return parsed === null ? '—' : parsed.toLocaleString(undefined, {maximumFractionDigits: 2}); };
  const setStatus = (message, kind = '') => { status.className = `status ${kind}`; status.textContent = message; };
  const setFlowStatus = (message, kind = '') => { $('flow-status').className = `status ${kind}`; $('flow-status').textContent = message; };
  const fileName = (path) => path.split('/').pop();
  const findFile = (name) => viewer.fileMap.get(name) || viewer.files.find((file) => file.name === fileName(name) || file.webkitRelativePath.endsWith(`/${name}`));
  const readText = async (name) => { const file = findFile(name); if (!file) throw new Error(`Missing ${name}`); return file.text(); };
  const readJson = async (name) => JSON.parse(await readText(name));

  function parseSmallCsv(source) {
    const rows = []; let row = []; let cell = ''; let quoted = false;
    for (let i = 0; i < source.length; i += 1) {
      const char = source[i];
      if (char === '"') { if (quoted && source[i + 1] === '"') { cell += '"'; i += 1; } else quoted = !quoted; }
      else if (char === ',' && !quoted) { row.push(cell); cell = ''; }
      else if ((char === '\n' || char === '\r') && !quoted) { if (char === '\r' && source[i + 1] === '\n') i += 1; row.push(cell); cell = ''; if (row.some((item) => item !== '')) rows.push(row); row = []; }
      else cell += char;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const headers = rows.shift() || [];
    return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
  }

  function idbRequest(request) {
    return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
  }
  async function openCache() {
    if (viewer.cacheDb) return viewer.cacheDb;
    viewer.cacheDb = await new Promise((resolve, reject) => {
      const request = indexedDB.open('gravity-viewer-cache', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('partitions', {keyPath: 'key'});
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    return viewer.cacheDb;
  }
  async function cacheGet(key) { try { const db = await openCache(); return idbRequest(db.transaction('partitions').objectStore('partitions').get(key)); } catch (_) { return null; } }
  async function cachePut(record) { try { const db = await openCache(); await idbRequest(db.transaction('partitions', 'readwrite').objectStore('partitions').put(record)); } catch (_) { setFlowStatus('Browser storage quota or IndexedDB is unavailable; continuing without persistent cache.', ''); } }
  async function clearCache() { try { const db = await openCache(); await idbRequest(db.transaction('partitions', 'readwrite').objectStore('partitions').clear()); setFlowStatus('Viewer cache cleared.'); } catch (_) { setFlowStatus('Viewer cache could not be cleared.', 'bad'); } }

  async function sha256(file) {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  }
  async function validatePartition(file, descriptor) {
    if (!file) throw new Error(`Partition file is missing: ${descriptor.path || descriptor.file}`);
    const expectedSize = descriptor.size_bytes ?? descriptor.size;
    if (expectedSize !== undefined && file.size !== expectedSize) throw new Error(`Partition size mismatch: ${descriptor.path || descriptor.file}`);
    if (descriptor.sha256) {
      const actual = await sha256(file);
      if (actual !== descriptor.sha256) throw new Error(`Partition checksum mismatch: ${descriptor.path || descriptor.file}`);
    }
  }
  async function readPartition(kind, bin, filters = {}, limit = 5000) {
    const group = viewer.summary[kind];
    if (!group?.available) return [];
    const descriptor = (group.partitions || []).find((item) => item.key === bin) || (bin === 'all' ? null : undefined);
    const descriptors = descriptor ? [descriptor] : (bin === 'all' ? (group.partitions || []) : []);
    const output = [];
    for (const item of descriptors) {
      const path = item.path || item.file; const file = findFile(path); await validatePartition(file, item);
      const identity = `${viewer.cacheIdentity}|${kind}|${path}|${item.sha256 || file.size}|${JSON.stringify(filters)}|v${CACHE_VERSION}`;
      const cacheKey = identity;
      const cached = await cacheGet(cacheKey);
      if (cached && Array.isArray(cached.rows)) { output.push(...cached.rows); continue; }
      const rows = await new Promise((resolve, reject) => {
        const worker = new Worker(workerUrl); const collected = [];
        worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
        worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
        worker.postMessage({file, filters, limit: Math.max(limit, 5000), chunkSize: 1024 * 1024});
      });
      await cachePut({key: cacheKey, identity, partition: path, rows}); output.push(...rows);
    }
    return output.slice(0, limit);
  }

  function renderTable(rows, title = '') {
    const target = $('flow-table');
    if (!rows.length) { target.querySelector('thead').innerHTML = ''; target.querySelector('tbody').innerHTML = `<tr><td class="muted">No rows for this selection.</td></tr>`; return; }
    const headers = Object.keys(rows[0]);
    target.querySelector('thead').innerHTML = `<tr>${headers.map((header) => `<th>${esc(header)}</th>`).join('')}</tr>`;
    target.querySelector('tbody').innerHTML = rows.slice(0, 500).map((row) => `<tr>${headers.map((header) => `<td>${esc(row[header])}</td>`).join('')}</tr>`).join('');
    $('flow-detail').innerHTML = `<dl><dt>View</dt><dd>${esc(title)}</dd><dt>Rows shown</dt><dd>${display(rows.length)}</dd><dt>Display bound</dt><dd>500 rows; file parsing is streamed in a worker.</dd></dl>`;
    drawFlowMap(rows, title);
  }
  function options(select, values, placeholder = 'All') { select.innerHTML = `<option value="">${esc(placeholder)}</option>${[...new Set(values.filter(Boolean))].sort().map((value) => `<option value="${esc(value)}">${esc(value)}</option>`).join('')}`; }
  function canonicalBins(values) {
    return [...new Set(values || [])].sort((a, b) => {
      const ai = Number(String(a).match(/\d+/)?.[0]); const bi = Number(String(b).match(/\d+/)?.[0]);
      if (Number.isFinite(ai) && Number.isFinite(bi) && ai !== bi) return ai - bi;
      return String(a).localeCompare(String(b));
    });
  }
  function populateFlowSelectors() {
    const od = viewer.summary.od || {}; const route = viewer.summary.route_flows || {};
    options($('flow-bin'), canonicalBins(od.bins || []), 'Select time slot');
    options($('flow-origin'), od.origins || [], 'All origins'); options($('flow-destination'), od.destinations || [], 'All destinations');
    options($('flow-line'), route.lines || [], 'All lines'); options($('flow-trip'), route.trips || [], 'All trips');
    options($('flow-transfers'), route.transfer_counts || ['0', '1', '2', '3'], 'All transfers');
    if ((od.bins || []).length) $('flow-bin').value = od.bins[0];
    const definition = route.transfer_count_definition || 'No realized route-assignment transfer count is present.';
    $('flow-semantics').textContent = route.available ? `Estimated OD demand is separate from route-assignment flow. ${definition}. A service total is expected assigned demand or passenger-leg usage, not unique passengers when a journey uses multiple legs.` : `${definition} Existing bundles without route-flow partitions support the OD destination view only.`;
  }

  function sectionSummary() { return viewer.summary.route_sections || {}; }
  function selectedSectionLines() {
    return [...($('section-lines')?.selectedOptions || [])].map((option) => option.value).filter(Boolean);
  }
  function populateSectionControls() {
    const summary = sectionSummary(); const panel = $('route-sections');
    if (!panel) return;
    panel.classList.remove('hidden');
    if (!summary.available) {
      $('section-unavailable').textContent = summary.message || 'Route-section data unavailable. This bundle still supports OD destination exploration.';
      $('section-unavailable').classList.remove('hidden');
      [...panel.querySelectorAll('button, select, input')].forEach((element) => { if (element.id !== 'section-clear-cache') element.disabled = true; });
      return;
    }
    const route = viewer.summary.route_flows || {};
    const lineSelect = $('section-lines');
    lineSelect.innerHTML = (route.lines || []).map((line) => `<option value="${esc(line)}" selected>${esc(line)}</option>`).join('');
    options($('section-pattern'), route.patterns || [], 'All patterns'); options($('section-direction'), route.directions || [], 'All directions');
    const bins = canonicalBins((viewer.summary.od || {}).bins || []); const slider = $('section-bin');
    slider.max = Math.max(0, bins.length - 1); slider.value = 0; slider.dataset.bins = JSON.stringify(bins); updateSectionBinLabel();
    const catalogFile = findFile(summary.catalog_path); if (catalogFile) catalogFile.text().then((source) => { const rows = parseSmallCsv(source); $('section-id').innerHTML = '<option value="">Choose a directed section</option>' + rows.map((row) => `<option value="${esc(row.section_id)}">${esc(row.line_id)} / ${esc(row.from_stop_id)} → ${esc(row.to_stop_id)}</option>`).join(''); }).catch(() => {});
    refreshSectionLineChips();
  }
  function sectionBins() { try { return JSON.parse($('section-bin').dataset.bins || '[]'); } catch (_) { return []; } }
  function updateSectionBinLabel() { const bins = sectionBins(); const value = bins[Number($('section-bin')?.value || 0)] || '—'; $('section-bin-label').textContent = value; }
  function refreshSectionLineChips() { $('section-line-chips').innerHTML = selectedSectionLines().map((line) => `<span class="chip">${esc(line)}</span>`).join('') || '<span class="muted small">No line filter</span>'; applySectionLineFilter(); }
  function applySectionLineFilter() {
    if (!viewer.sectionLayer) return;
    const selected = new Set(selectedSectionLines());
    viewer.sectionLayer.eachLayer((layer) => { const line = layer.feature?.properties?.line_id; const visible = selected.has(line); if (visible) layer.addTo(viewer.sectionLayer); else viewer.sectionLayer.removeLayer(layer); });
    // Re-add hidden layers only when the filter changes, without rebuilding geometry.
    viewer.sectionFeatures.forEach((feature) => { if (!visibleSectionFeature(feature, selected)) return; if (!viewer.sectionLayer.hasLayer(feature._layer)) feature._layer.addTo(viewer.sectionLayer); });
  }
  function visibleSectionFeature(feature, selected) { return selected.has(feature.properties?.line_id); }
  async function readSectionPartition(groupName, bin, filters = {}, limit = 500, offset = 0) {
    const group = sectionSummary()[groupName] || {}; const descriptor = (group.partitions || []).find((item) => String(item.key) === String(bin));
    if (!descriptor) return {rows: [], cached: false};
    const path = descriptor.path || descriptor.file; const file = findFile(path); await validatePartition(file, descriptor);
    const identity = `${viewer.cacheIdentity}|route_sections|${groupName}|${path}|${descriptor.sha256 || file.size}|${bin}|${offset}|${limit}|${JSON.stringify(filters)}|v${CACHE_VERSION}`;
    const cached = await cacheGet(identity);
    if (cached && Array.isArray(cached.rows)) return {rows: cached.rows, total: cached.total ?? cached.rows.length, cached: true};
    let total = 0;
    const rows = await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const collected = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { total = event.data.total ?? collected.length; worker.terminate(); resolve(collected); } };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
      worker.postMessage({file, filters, offset, limit, scanAll: true, chunkSize: 1024 * 1024});
    });
    await cachePut({key: identity, identity, partition: path, rows, total}); return {rows, total, cached: false};
  }
  async function inspectSection(feature) {
    const summary = sectionSummary(); if (!summary.available) return;
    const bins = sectionBins(); const bin = bins[Number($('section-bin').value || 0)]; const properties = feature.properties || feature || {};
    const filters = {section_id: properties.section_id}; const lines = selectedSectionLines(); if (lines.length === 1) filters.line_id = lines[0];
    if ($('section-pattern').value) filters.pattern_id = $('section-pattern').value; if ($('section-direction').value) filters.direction = $('section-direction').value;
    $('section-status').textContent = `Loading section ${properties.section_id} for ${bin}…`;
    try {
      const load = await readSectionPartition('section_loads', bin, filters, 1000); const contributors = await readSectionPartition('contributors', bin, filters, viewer.sectionPageSize, viewer.sectionPage * viewer.sectionPageSize);
      viewer.selectedSection = properties; const loadRow = load.rows[0] || {}; const distinct = loadRow.distinct_od_count || new Set(contributors.rows.map((row) => `${row.origin_stop_id}|${row.destination_stop_id}`)).size;
      $('section-detail').innerHTML = `<dl><dt>Section</dt><dd>${esc(properties.line_id)} / ${esc(properties.pattern_id)}: ${esc(properties.from_stop_id)} → ${esc(properties.to_stop_id)}</dd><dt>Interval</dt><dd>${esc(bin)} (OD departure interval)</dd><dt>Assigned section flow</dt><dd>${display(loadRow.assigned_flow || contributors.rows.reduce((sum, row) => sum + (num(row.assigned_demand) || 0), 0))}</dd><dt>Distinct ODs</dt><dd>${esc(distinct)}</dd><dt>Meaning</dt><dd>Saved path contribution per traversed section; transfer journeys may contribute on multiple legs.</dd></dl>`;
      renderSectionTable(contributors.rows); $('section-page-label').textContent = `Page ${viewer.sectionPage + 1} of ${Math.max(1, Math.ceil(contributors.total / viewer.sectionPageSize))} (${contributors.total} matching ODs/contributions)`; $('section-status').textContent = `${contributors.cached ? 'Cache hit' : 'Read selected partition'}: selected interval/section only.`;
    } catch (error) { $('section-status').textContent = `Could not load section: ${error.message}`; }
  }
  function chooseSection(feature) { viewer.sectionPage = 0; inspectSection(feature); }
  function sectionClick(feature) {
    const properties = feature.properties || {}; const candidates = viewer.sectionFeatures.filter((item) => {
      const other = item.properties || {};
      return other.from_stop_id === properties.from_stop_id && other.to_stop_id === properties.to_stop_id;
    });
    if (candidates.length > 1) {
      $('section-detail').innerHTML = `<p>Multiple directed patterns overlap this stop pair. Choose one:</p>${candidates.map((item, index) => `<button type="button" data-section-choice="${index}">${esc(item.properties?.line_id)} / ${esc(item.properties?.pattern_id)} / ${esc(item.properties?.direction || '—')}</button>`).join(' ')}`;
      candidates.forEach((item, index) => $('section-detail').querySelector(`[data-section-choice="${index}"]`)?.addEventListener('click', () => chooseSection(item)));
      return;
    }
    chooseSection(feature);
  }
  async function loadSectionSummary() {
    if (!sectionSummary().available) return;
    const bins = sectionBins(); const bin = bins[Number($('section-bin').value || 0)];
    try {
      const summary = await readSectionPartition('section_loads', bin, {}, 100000);
      $('section-status').textContent = `${summary.cached ? 'Cache hit' : 'Read selected interval'}: ${summary.total} directed section loads for ${bin}. Contributors remain unopened until a section is selected.`;
    } catch (error) { $('section-status').textContent = `Could not load section summary: ${error.message}`; }
  }
  function renderSectionTable(rows) {
    const target = $('section-table'); if (!rows.length) { target.querySelector('thead').innerHTML = ''; target.querySelector('tbody').innerHTML = '<tr><td class="muted">No matching OD contributors.</td></tr>'; return; }
    const headers = Object.keys(rows[0]); target.querySelector('thead').innerHTML = `<tr>${headers.map((header) => `<th>${esc(header)}</th>`).join('')}</tr>`;
    target.querySelector('tbody').innerHTML = rows.slice(0, 500).map((row) => `<tr>${headers.map((header) => `<td>${esc(row[header])}</td>`).join('')}</tr>`).join('');
  }

  async function refreshFlows() {
    const bin = $('flow-bin').value || 'all'; const limit = Math.max(1, Math.min(500, Number($('flow-limit').value) || 50));
    const kind = $('flow-kind')?.value || 'od'; const filters = {};
    if (kind === 'od') { if ($('flow-origin').value) filters.origin_stop_id = $('flow-origin').value; if ($('flow-destination').value) filters.destination_stop_id = $('flow-destination').value; }
    else { if ($('flow-line').value) filters.line_id = $('flow-line').value; if ($('flow-trip').value) filters.trip_id = $('flow-trip').value; if ($('flow-origin').value) filters.origin_stop_id = $('flow-origin').value; if ($('flow-destination').value) filters.destination_stop_id = $('flow-destination').value; if ($('flow-transfers').value) filters.transfer_count = $('flow-transfers').value; }
    setFlowStatus(`Loading ${kind === 'od' ? 'estimated OD demand' : 'route-assignment flows'} for ${bin}…`);
    try {
      const rows = await readPartition(kind === 'od' ? 'od' : 'route_flows', bin, filters, limit * 4);
      if (kind === 'od') {
        rows.sort((a, b) => (num(b.estimated_demand) || 0) - (num(a.estimated_demand) || 0));
      } else rows.sort((a, b) => (num(b.assigned_demand) || 0) - (num(a.assigned_demand) || 0));
      viewer.odRows = kind === 'od' ? rows.slice(0, limit) : []; viewer.routeRows = kind === 'route' ? rows.slice(0, limit) : [];
      renderTable(rows.slice(0, limit), kind === 'od' ? 'Estimated OD demand' : 'Route-assignment flow');
      setFlowStatus(`Loaded ${Math.min(rows.length, limit)} bounded rows from ${bin === 'all' ? 'the selected partitions' : 'one time partition'}.`, 'ok');
    } catch (error) { setFlowStatus(`Could not load partition: ${error.message}`, 'bad'); }
  }
  // Kept as named actions for integrations and the legacy map controls.
  function selectOd(row) { viewer.selectedOd = row; }
  function chooseOdStop(id) { if (!$('flow-origin').value) $('flow-origin').value = id; else $('flow-destination').value = id; refreshFlows(); }

  function drawFlowMap(rows, title) {
    if (!viewer.map || !window.L) return;
    if (!viewer.flowLayer) viewer.flowLayer = L.layerGroup().addTo(viewer.map); else viewer.flowLayer.clearLayers();
    rows.slice(0, 100).forEach((row) => {
      const originId = title === 'Estimated OD demand' ? row.origin_stop_id : row.boarding_stop_id || row.origin_stop_id;
      const destinationId = title === 'Estimated OD demand' ? row.destination_stop_id : row.alighting_stop_id || row.destination_stop_id;
      const origin = viewer.stopsById.get(originId); const destination = viewer.stopsById.get(destinationId);
      if (!origin || !destination) return;
      const a = [num(origin.lat), num(origin.lon)]; const b = [num(destination.lat), num(destination.lon)];
      if (a.some((value) => value === null) || b.some((value) => value === null)) return;
      L.polyline([a, b], {color: title === 'Estimated OD demand' ? '#235aa6' : '#d14b4b', weight: 2, opacity: .45}).bindTooltip(`${esc(originId)} → ${esc(destinationId)}`).addTo(viewer.flowLayer);
    });
  }

  async function waitForLeaflet() { for (let i = 0; i < 100 && !window.L; i += 1) await new Promise((resolve) => setTimeout(resolve, 50)); if (!window.L) throw new Error('Map library could not be loaded.'); }
  async function renderNetwork() {
    const stopsFile = findFile('network/stops.csv') || findFile('stops.csv'); if (!stopsFile) return;
    viewer.stops = await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const rows = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') rows.push(...event.data.rows); else { worker.terminate(); resolve(rows); } };
      worker.onerror = () => { worker.terminate(); reject(new Error('network CSV worker failed')); };
      worker.postMessage({file: stopsFile, filters: {}, limit: 20000, chunkSize: 1024 * 1024});
    });
    viewer.stopsById = new Map(viewer.stops.map((stop) => [stop.stop_id, stop]));
    try { await waitForLeaflet(); } catch (error) { $('network-map').classList.remove('hidden'); $('map').textContent = `${error.message} Tables remain available without map tiles.`; return; }
    const points = viewer.stops.map((stop) => [num(stop.lat), num(stop.lon)]).filter(([lat, lon]) => lat !== null && lon !== null); if (!points.length) return;
    $('network-map').classList.remove('hidden'); viewer.map = L.map('map'); L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(viewer.map); viewer.map.fitBounds(L.latLngBounds(points).pad(0.08)); viewer.markerLayer = L.layerGroup().addTo(viewer.map); viewer.flowLayer = L.layerGroup().addTo(viewer.map);
    viewer.stops.forEach((stop) => { const lat = num(stop.lat); const lon = num(stop.lon); if (lat === null || lon === null) return; L.circleMarker([lat, lon], {radius: 4, color: '#235aa6', fillOpacity: .7}).bindTooltip(`${esc(stop.stop_id)}: ${esc(stop.name || '')}`).on('click', () => chooseOdStop(stop.stop_id)).addTo(viewer.markerLayer); });
    await renderLineGeometry();
  }

  async function renderLineGeometry() {
    const geometry = viewer.summary.line_geometry || viewer.summary.route_sections || {};
    if (!geometry.available || !geometry.geometry_path || !viewer.map || !window.L) return;
    const file = findFile(geometry.geometry_path); if (!file) return;
    try {
      const geojson = JSON.parse(await file.text()); viewer.sectionLayer = L.layerGroup().addTo(viewer.map);
      viewer.sectionFeatures = (geojson.features || []).map((feature) => {
        const selected = L.geoJSON(feature, {style: {color: '#6a4fb3', weight: 4, opacity: .75}}); const layer = selected.getLayers()[0];
        feature._layer = layer; layer.feature = feature; layer.on('click', () => sectionClick(feature)); layer.bindTooltip(`${feature.properties?.line_id || ''}: ${feature.properties?.from_stop_id || ''} → ${feature.properties?.to_stop_id || ''}`); layer.addTo(viewer.sectionLayer); return feature;
      });
    } catch (error) { setFlowStatus(`Line geometry unavailable: ${error.message}`, ''); }
  }

  async function validateStartup(manifest) {
    const problems = [];
    for (const name of REQUIRED) if (!findFile(name)) problems.push(`missing required file: ${name}`);
    for (const [name, descriptor] of Object.entries(manifest.files || {})) { const file = findFile(name); if (!file) { problems.push(`missing manifest file: ${name}`); continue; } const size = descriptor.size_bytes ?? descriptor.size; if (size !== undefined && file.size !== size) problems.push(`size mismatch: ${name}`); }
    return problems;
  }
  async function load(files) {
    try {
      viewer.files = files; viewer.fileMap = new Map(files.map((file) => [file.webkitRelativePath || file.name, file])); setStatus('Reading manifest and small summary; large tables remain unopened…');
      viewer.manifest = await readJson('bundle_manifest.json'); if (viewer.manifest.bundle_type !== 'gravity_viewer_bundle') throw new Error(`Unexpected bundle type: ${viewer.manifest.bundle_type}`);
      const problems = await validateStartup(viewer.manifest); viewer.model = await readJson('model_specification.json'); viewer.report = await readJson('report.json'); viewer.summary = findFile('viewer_summary.json') ? await readJson('viewer_summary.json') : {};
      if (!viewer.summary.od) {
        const legacyRecord = viewer.manifest.files?.['full_od.csv'] || {};
        viewer.summary = {
          od: {available: true, partition_field: 'legacy_full_table', bins: ['all'], origins: [], destinations: [], partitions: [{key: 'all', path: 'full_od.csv', size_bytes: legacyRecord.size_bytes, sha256: legacyRecord.sha256}]},
          route_flows: {available: false, message: 'Legacy bundle has no route-flow partitions.'},
        };
      }
      viewer.cacheIdentity = `${viewer.manifest.model_fingerprint || ''}|${viewer.manifest.specification_fingerprint || ''}|${viewer.manifest.report_fingerprint || ''}|${JSON.stringify(viewer.manifest.provenance || {})}`;
      $('summary').classList.remove('hidden'); $('diagnostics').classList.remove('hidden'); $('tables').classList.remove('hidden'); $('flows').classList.remove('hidden');
      const counts = viewer.manifest.row_counts || {}; $('metrics').innerHTML = [metric('Bundle type', viewer.manifest.bundle_type), metric('Schema', viewer.manifest.schema_version), metric('OD rows', counts['full_od.csv'] ?? 'unknown'), metric('Measurements', counts['predicted_measurements.csv'] ?? 'unknown'), metric('OD partitions', viewer.summary.od?.partitions?.length ?? 'legacy'), metric('Route flows', viewer.summary.route_flows?.available ? 'available' : 'absent'), metric('Identifiability', viewer.manifest.identifiability?.available ? 'available' : 'not available'), metric('Network files', Object.keys(viewer.manifest.files || {}).filter((key) => key.startsWith('network/')).length)].join('');
      const modelLikelihood = viewer.model.likelihood || viewer.model.specification?.likelihood?.family; $('model').innerHTML = renderKeyValue({fingerprint: viewer.manifest.model_fingerprint, likelihood: modelLikelihood}); $('provenance').innerHTML = renderKeyValue(viewer.manifest.provenance || viewer.manifest.fit_provenance); const adequacy = viewer.report.adequacy || {}; $('diagnostic-metrics').innerHTML = [metric('Observed total', adequacy.observed_total), metric('Modeled total', adequacy.modeled_total), metric('MAE', adequacy.mae), metric('RMSE', adequacy.rmse), metric('Report fingerprint', viewer.report.report_fingerprint, 'fingerprint')].join('');
      $('table-content').innerHTML = '<p class="muted">Large tables are available through the partitioned explorer. No complete CSV is parsed at startup.</p>';
      const executive = findFile('executive_summary.md'); $('executive').innerHTML = executive ? `<pre>${esc(await executive.text())}</pre>` : '<p>No executive summary supplied.</p>'; populateFlowSelectors(); populateSectionControls(); await renderNetwork(); if (viewer.summary.route_sections?.available) await loadSectionSummary(); else await refreshFlows();
      if (!viewer.summary.od) { $('interactive-notice').textContent = 'This is a legacy bundle without partition metadata. The validated tables remain available; route-flow and lazy time-slot exploration require a refreshed export.'; $('interactive-notice').classList.remove('hidden'); }
      if (problems.length) setStatus(`Bundle loaded with startup validation warnings:\n- ${problems.join('\n- ')}`, 'bad'); else setStatus('Bundle manifest and summary loaded. Partition checksums are verified when selected.', 'ok');
    } catch (error) { $('summary').classList.add('hidden'); $('diagnostics').classList.add('hidden'); $('flows').classList.add('hidden'); setStatus(`Could not load bundle: ${error.message}`, 'bad'); }
  }
  function metric(label, value, extra = '') { return `<div class="metric"><label>${esc(label)}</label><strong class="${extra}">${esc(value)}</strong></div>`; }
  function renderKeyValue(value) { if (!value || typeof value !== 'object') return `<p>${esc(value)}</p>`; return `<div class="fingerprint">${Object.entries(value).map(([key, item]) => `<div><b>${esc(key)}:</b> ${esc(typeof item === 'object' ? JSON.stringify(item) : item)}</div>`).join('')}</div>`; }
  $('folder').addEventListener('change', (event) => load([...event.target.files])); $('flow-refresh').addEventListener('click', refreshFlows); $('flow-clear-cache').addEventListener('click', clearCache); ['flow-bin', 'flow-origin', 'flow-destination', 'flow-line', 'flow-trip', 'flow-transfers', 'flow-kind'].forEach((id) => { if ($(id)) $(id).addEventListener('change', refreshFlows); });
  $('section-lines')?.addEventListener('change', refreshSectionLineChips); $('section-all-lines')?.addEventListener('click', () => { [...$('section-lines').options].forEach((option) => { option.selected = true; }); refreshSectionLineChips(); }); $('section-clear-lines')?.addEventListener('click', () => { [...$('section-lines').options].forEach((option) => { option.selected = false; }); refreshSectionLineChips(); });
  $('section-pattern')?.addEventListener('change', () => viewer.selectedSection && inspectSection(viewer.selectedSection)); $('section-direction')?.addEventListener('change', () => viewer.selectedSection && inspectSection(viewer.selectedSection)); $('section-bin')?.addEventListener('input', updateSectionBinLabel); $('section-bin')?.addEventListener('change', () => { updateSectionBinLabel(); viewer.sectionPage = 0; loadSectionSummary(); if (viewer.selectedSection) inspectSection(viewer.selectedSection); }); $('section-previous')?.addEventListener('click', () => { const slider = $('section-bin'); slider.value = Math.max(0, Number(slider.value) - 1); slider.dispatchEvent(new Event('change')); }); $('section-next')?.addEventListener('click', () => { const slider = $('section-bin'); slider.value = Math.min(Number(slider.max), Number(slider.value) + 1); slider.dispatchEvent(new Event('change')); }); $('section-page-previous')?.addEventListener('click', () => { viewer.sectionPage = Math.max(0, viewer.sectionPage - 1); if (viewer.selectedSection) inspectSection(viewer.selectedSection); }); $('section-page-next')?.addEventListener('click', () => { viewer.sectionPage += 1; if (viewer.selectedSection) inspectSection(viewer.selectedSection); }); $('section-clear-cache')?.addEventListener('click', clearCache);
  $('section-inspect')?.addEventListener('click', () => { const id = $('section-id').value; const feature = viewer.sectionFeatures.find((item) => item.properties?.section_id === id); if (feature) chooseSection(feature); else if (id) inspectSection({section_id: id}); });
})();
