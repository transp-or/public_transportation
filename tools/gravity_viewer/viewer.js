/* Memory-bounded gravity viewer.
 *
 * The manifest and viewer_summary.json are intentionally the only large-scale
 * metadata read during startup. CSV partitions are parsed in a Web Worker and
 * cached per bundle identity/time slot in IndexedDB. The viewer never builds a
 * JavaScript array for the complete OD table.
 */
(() => {
  'use strict';

  const CACHE_VERSION = 3;
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
      const {file, filters, limit, chunkSize, offset = 0, scanAll = false, rankBy = null, rankAbs = false, streamMatches = false} = event.data;
      let header = null; let carry = ''; let matched = 0; let sent = 0; const ranked = [];
      let streamed = [];
      const postStreamBatch = async () => {
        if (!streamed.length) return;
        const rows = streamed; streamed = [];
        await new Promise((resolve) => { self.onmessage = () => resolve(); self.postMessage({type: 'rows', rows}); });
      };
      const accept = (row) => Object.entries(filters || {}).every(([key, value]) => Array.isArray(value) ? value.map(String).includes(String(row[key] ?? '')) : String(row[key] ?? '') === String(value));
      const rank = (row) => { const value = Number(row[rankBy]); return rankAbs ? Math.abs(value) : value; };
      const siftUp = (index) => {
        while (index > 0) {
          const parent = Math.floor((index - 1) / 2);
          if (ranked[parent].score <= ranked[index].score) break;
          [ranked[parent], ranked[index]] = [ranked[index], ranked[parent]];
          index = parent;
        }
      };
      const siftDown = (index) => {
        while (true) {
          const left = index * 2 + 1; const right = left + 1;
          let smallest = index;
          if (left < ranked.length && ranked[left].score < ranked[smallest].score) smallest = left;
          if (right < ranked.length && ranked[right].score < ranked[smallest].score) smallest = right;
          if (smallest === index) break;
          [ranked[index], ranked[smallest]] = [ranked[smallest], ranked[index]];
          index = smallest;
        }
      };
      const retain = (row) => {
        const score = rank(row); if (!Number.isFinite(score)) return;
        const candidate = {row, score};
        if (ranked.length < limit) { ranked.push(candidate); siftUp(ranked.length - 1); return; }
        if (score > ranked[0].score) { ranked[0] = candidate; siftDown(0); }
      };
      const emit = async (lines, final = false) => {
        const rows = [];
        for (const line of lines) {
          if (!line) continue;
          const cells = parseLine(line);
          if (!header) { header = cells; continue; }
          const row = Object.fromEntries(header.map((name, index) => [name, cells[index] ?? '']));
          if (accept(row)) {
            if (streamMatches) { streamed.push(row); matched += 1; if (streamed.length >= 512) await postStreamBatch(); }
            else if (rankBy) { retain(row); matched += 1; }
            else { if (matched >= offset && sent < limit) { rows.push(row); sent += 1; } matched += 1; }
          }
        }
        if (rows.length) self.postMessage({type: 'rows', rows});
        if (streamMatches && final && streamed.length) await postStreamBatch();
        if (final && !rankBy && !streamMatches) self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
      };
      for (let offset = 0; offset < file.size && (scanAll || rankBy || sent < limit); offset += chunkSize) {
        carry += await file.slice(offset, Math.min(file.size, offset + chunkSize)).text();
        const lines = carry.split(/\\r?\\n/); carry = lines.pop() || ''; await emit(lines);
      }
      if (carry) await emit([carry], true);
      if (streamMatches) {
        self.postMessage({type: 'done', truncated: false, total: matched});
      } else if (!carry && !rankBy) self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
      if (rankBy) { ranked.sort((a, b) => b.score - a.score); self.postMessage({type: 'rows', rows: ranked.map((item) => item.row)}); self.postMessage({type: 'done', truncated: matched > ranked.length, total: matched}); }
    };
  `;
  const workerUrl = URL.createObjectURL(new Blob([workerSource], {type: 'text/javascript'}));

  const viewer = {
    files: [], fileMap: new Map(), manifest: null, summary: {}, model: {}, report: {},
    map: null, stops: [], stopsById: new Map(), markerLayer: null, selectedOd: null,
    cacheDb: null, cacheIdentity: '', odRows: [], routeRows: [],
    mapClickMode: 'od', measurementLocationManifest: null, selectedMeasurementStop: '',
    selectedMeasurement: null, measurementRows: [], measurementRowPage: 0, measurementRowsPerPage: 50,
    measurementContributionPage: 0, measurementContributions: new Map(), measurementQueryId: 0,
    measurementQueryTimer: null,
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
      const request = indexedDB.open('gravity-viewer-cache', 2);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains('partitions')) db.createObjectStore('partitions', {keyPath: 'key'});
        if (!db.objectStoreNames.contains('measurementContributions')) {
          const store = db.createObjectStore('measurementContributions', {keyPath: ['identity', 'measurement_index', 'rank', 'od_index']});
          store.createIndex('byMeasurement', ['identity', 'measurement_index'], {unique: false});
          store.createIndex('byRank', ['identity', 'measurement_index', 'rank', 'od_index'], {unique: true});
        }
      };
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    return viewer.cacheDb;
  }
  async function cacheGet(key) { try { const db = await openCache(); return idbRequest(db.transaction('partitions').objectStore('partitions').get(key)); } catch (_) { return null; } }
  async function cachePut(record) { try { const db = await openCache(); await idbRequest(db.transaction('partitions', 'readwrite').objectStore('partitions').put(record)); } catch (_) { setFlowStatus('Browser storage quota or IndexedDB is unavailable; continuing without persistent cache.', ''); } }
  async function clearCache() { try { const db = await openCache(); const tx = db.transaction(['partitions', 'measurementContributions'], 'readwrite'); tx.objectStore('partitions').clear(); tx.objectStore('measurementContributions').clear(); await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error); }); viewer.measurementContributions.clear(); setFlowStatus('Viewer cache cleared.'); } catch (_) { setFlowStatus('Viewer cache could not be cleared.', 'bad'); } }

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
  async function readPartition(kind, bin, filters = {}, limit = 5000, options = {}) {
    const group = viewer.summary[kind];
    if (!group?.available) return [];
    const matching = (group.partitions || []).filter((item) => item.key === bin);
    const descriptors = bin === 'all'
      ? (group.partitions || [])
      : (kind === 'measurement_od_contributions' ? matching : matching.slice(0, 1));
    const output = [];
    const rankOutput = () => {
      output.sort((a, b) => {
        const left = Math.abs(Number(a[options.rankBy]) || 0);
        const right = Math.abs(Number(b[options.rankBy]) || 0);
        return right - left;
      });
      if (output.length > limit) output.length = limit;
    };
    for (const item of descriptors) {
      const path = item.path || item.file; const file = findFile(path); await validatePartition(file, item);
      const identity = `${viewer.cacheIdentity}|${kind}|${path}|${item.sha256 || file.size}|${JSON.stringify(filters)}|${JSON.stringify(options)}|${limit}|v${CACHE_VERSION}`;
      const cacheKey = identity;
      const cached = await cacheGet(cacheKey);
      if (cached && Array.isArray(cached.rows)) {
        output.push(...cached.rows);
        if (options.rankBy) rankOutput();
        continue;
      }
      const rows = await new Promise((resolve, reject) => {
        const worker = new Worker(workerUrl); const collected = [];
        worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
        worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
        worker.postMessage({file, filters, limit: Math.max(limit, 1), chunkSize: 1024 * 1024, ...options});
      });
      await cachePut({key: cacheKey, identity, partition: path, rows}); output.push(...rows);
      if (options.rankBy) rankOutput();
    }
    return output.slice(0, limit);
  }

  async function readDescriptorPage(descriptor, filters, offset, limit, cacheScope) {
    const path = descriptor.path || descriptor.file; const file = findFile(path);
    await validatePartition(file, descriptor);
    const cacheKey = `${viewer.cacheIdentity}|${cacheScope}|${path}|${JSON.stringify(filters)}|${offset}|${limit}|v${CACHE_VERSION}`;
    const cached = await cacheGet(cacheKey);
    if (cached && Array.isArray(cached.rows)) return cached.rows;
    const rows = await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const collected = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
      worker.postMessage({file, filters, limit, offset, chunkSize: 1024 * 1024});
    });
    await cachePut({key: cacheKey, identity: cacheKey, partition: path, rows});
    return rows;
  }

  async function measurementLocationIndex() {
    const summary = viewer.summary.measurement_locations || {};
    if (!summary.available) return null;
    if (viewer.measurementLocationManifest) return viewer.measurementLocationManifest;
    if (typeof summary.manifest_path !== 'string' || !summary.manifest_path.startsWith('partitions/measurement_locations/') || summary.manifest_path.includes('..')) throw new Error('Measurement location lookup path is unsafe.');
    const file = findFile(summary.manifest_path);
    await validatePartition(file, {path: summary.manifest_path, size_bytes: undefined, sha256: summary.manifest_sha256});
    viewer.measurementLocationManifest = JSON.parse(await file.text());
    const index = viewer.measurementLocationManifest; const expected = Number(viewer.manifest.row_counts?.['predicted_measurements.csv']);
    if (index.schema_version !== 1 || index.dataset_type !== 'gravity_measurement_location_lookup' || !Number.isInteger(index.number_of_measurements) || index.number_of_measurements !== expected || index.number_of_stops !== Object.keys(index.stops || {}).length || index.number_of_measurements !== Number(index.indexed_measurements) + Number(index.missing_stop_rows)) throw new Error('Measurement location lookup schema or row counts are invalid.');
    if (!Array.isArray(index.columns) || !['measurement_index', 'measurement_type', 'line', 'direction', 'observation_time', 'time_period', 'stop', 'observed', 'modeled'].every((column, position) => index.columns[position] === column)) throw new Error('Measurement location lookup columns are invalid.');
    const source = index.source_identity || {}; const bundle = viewer.manifest;
    for (const [field, actual] of [['fit_fingerprint', bundle.model_fingerprint], ['assignment_fingerprint', bundle.assignment_fingerprint], ['canonical_index_fingerprint', bundle.canonical_index_fingerprint], ['binding_fingerprint', bundle.binding_fingerprint], ['od_layout_fingerprint', bundle.od_layout_fingerprint]]) if (actual != null && source[field] !== actual) throw new Error(`Measurement location lookup ${field} differs from bundle provenance.`);
    return viewer.measurementLocationManifest;
  }

  async function measurementsAtStop(stopId, page = 0, pageSize = 50) {
    const manifest = await measurementLocationIndex();
    if (!manifest) return {rows: [], total: 0};
    const stops = manifest.stops || {};
    if (!Object.prototype.hasOwnProperty.call(stops, stopId)) return {rows: [], total: 0};
    const entry = stops[stopId]; const total = Number(entry.row_count) || 0;
    const start = page * pageSize; const end = Math.min(total, start + pageSize); const rows = [];
    if (start >= end) return {rows, total};
    let partitionStart = 0;
    for (const descriptor of entry.partitions || []) {
      if (descriptor.key !== stopId || Number(descriptor.row_count) < 1 || Number(descriptor.row_count) > Number(manifest.partition_rows_limit) || !String(descriptor.path || '').startsWith('partitions/measurement_locations/') || String(descriptor.path || '').includes('..')) throw new Error('Measurement location partition descriptor is invalid.');
      const partitionEnd = partitionStart + Number(descriptor.row_count || 0);
      if (partitionEnd > start && partitionStart < end) {
        const localStart = Math.max(0, start - partitionStart); const localEnd = Math.min(Number(descriptor.row_count), end - partitionStart);
        const chunk = await readDescriptorPage(descriptor, {stop: stopId}, localStart, localEnd - localStart, 'measurement-location');
        rows.push(...chunk);
      }
      partitionStart = partitionEnd;
      if (partitionStart >= end) break;
    }
    if (rows.length !== end - start || rows.some((row) => row.stop !== stopId)) throw new Error('Measurement location lookup page is incomplete or inconsistent.');
    return {rows, total};
  }

  function measurementContributionMarker(measurementIndex) {
    return `${viewer.cacheIdentity}|measurement-contributions|${measurementIndex}|v${CACHE_VERSION}`;
  }

  async function writeMeasurementContributionBatch(measurementIndex, rows) {
    if (!rows.length) return;
    const db = await openCache(); const tx = db.transaction('measurementContributions', 'readwrite'); const store = tx.objectStore('measurementContributions');
    rows.forEach((row) => {
      const value = Number(row.modeled_mean_contribution); const odIndex = Number(row.od_index);
      if (!Number.isFinite(value) || !Number.isSafeInteger(odIndex)) throw new Error('Contribution partition contains an invalid value or OD index.');
      store.put({...row, identity: viewer.cacheIdentity, measurement_index: Number(measurementIndex), od_index: odIndex, rank: -Math.abs(value)});
    });
    await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error); });
  }

  async function cacheMeasurementContributions(measurementIndex) {
    const marker = measurementContributionMarker(measurementIndex); const cached = await cacheGet(marker);
    if (cached?.complete) return Number(cached.rowCount) || 0;
    if (viewer.measurementContributions.has(measurementIndex)) return viewer.measurementContributions.get(measurementIndex);
    const task = (async () => {
      const summary = viewer.summary.measurement_od_contributions || {};
      let total = 0;
      for (const descriptor of summary.partitions || []) {
        const path = descriptor.path || descriptor.file; const file = findFile(path);
        await validatePartition(file, descriptor);
        await new Promise((resolve, reject) => {
          const worker = new Worker(workerUrl);
          worker.onmessage = async (event) => {
            if (event.data.type === 'rows') {
              try { total += event.data.rows.length; await writeMeasurementContributionBatch(measurementIndex, event.data.rows); worker.postMessage({type: 'ack'}); }
              catch (error) { worker.terminate(); reject(error); }
            } else { worker.terminate(); resolve(); }
          };
          worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'Contribution CSV worker failed.')); };
          worker.postMessage({file, filters: {measurement_index: String(measurementIndex)}, scanAll: true, streamMatches: true, chunkSize: 1024 * 1024});
        });
      }
      await cachePut({key: marker, identity: marker, complete: true, rowCount: total});
      return total;
    })();
    viewer.measurementContributions.set(measurementIndex, task);
    try { return await task; } finally { viewer.measurementContributions.delete(measurementIndex); }
  }

  async function measurementContributionPage(measurementIndex, page, pageSize) {
    const rowCount = await cacheMeasurementContributions(measurementIndex);
    const db = await openCache(); const tx = db.transaction('measurementContributions', 'readonly'); const store = tx.objectStore('measurementContributions');
    const count = await idbRequest(store.index('byMeasurement').count(IDBKeyRange.only([viewer.cacheIdentity, Number(measurementIndex)])));
    if (count !== rowCount) throw new Error('Cached contribution index row count is inconsistent. Clear the viewer cache and retry.');
    const start = page * pageSize; const rows = [];
    if (start >= count) return {rows, total: count};
    const range = IDBKeyRange.bound([viewer.cacheIdentity, Number(measurementIndex), -Number.MAX_VALUE, 0], [viewer.cacheIdentity, Number(measurementIndex), 0, Number.MAX_SAFE_INTEGER]);
    await new Promise((resolve, reject) => {
      let skipped = 0; const request = store.index('byRank').openCursor(range);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor || rows.length >= pageSize) { resolve(); return; }
        if (skipped < start) skipped += 1;
        else { const {identity, rank, ...row} = cursor.value; rows.push(row); }
        cursor.continue();
      };
    });
    return {rows, total: count};
  }

  async function queryLocalMeasurementPage(measurementIndex, page, pageSize) {
    const summary = viewer.summary.measurement_od_contributions || {};
    const base = new URL($('measurement-query-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The on-demand query service must be an HTTP service on this computer.');
    }
    const endpoint = new URL('/api/measurement-od-contributions', base.origin);
    endpoint.searchParams.set('measurement_index', String(measurementIndex));
    endpoint.searchParams.set('page', String(page));
    endpoint.searchParams.set('page_size', String(pageSize));
    const response = await fetch(endpoint, {cache: 'no-store'});
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without a valid JSON response.`); }
    if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
    if (payload.fit_fingerprint !== summary.fit_fingerprint || payload.mapping_artifact_fingerprint !== summary.mapping_artifact_fingerprint) {
      throw new Error('Local query service fit or mapping identity differs from this viewer bundle.');
    }
    if (Number(payload.measurement_index) !== measurementIndex || !Array.isArray(payload.rows) || !Number.isSafeInteger(Number(payload.total))) {
      throw new Error('Local query service returned an invalid measurement page.');
    }
    return payload;
  }

  async function queryLocalOdPage(odIndex, pageSize) {
    const summary = viewer.summary.measurement_od_contributions || {};
    const base = new URL($('measurement-query-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The on-demand query service must be an HTTP service on this computer.');
    }
    const endpoint = new URL('/api/od-measurement-contributions', base.origin);
    endpoint.searchParams.set('od_index', String(odIndex));
    endpoint.searchParams.set('page', '0');
    endpoint.searchParams.set('page_size', String(pageSize));
    const response = await fetch(endpoint, {cache: 'no-store'});
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without a valid JSON response.`); }
    if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
    if (payload.fit_fingerprint !== summary.fit_fingerprint || payload.mapping_artifact_fingerprint !== summary.mapping_artifact_fingerprint) {
      throw new Error('Local query service fit or mapping identity differs from this viewer bundle.');
    }
    if (Number(payload.od_index) !== odIndex || !Array.isArray(payload.rows) || !Number.isSafeInteger(Number(payload.total))) {
      throw new Error('Local query service returned an invalid OD contribution page.');
    }
    return payload;
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
    const semantics = route.flow_semantics || [];
    const serviceMarginal = semantics.includes('service_marginal');
    $('flow-semantics').textContent = route.available
      ? (serviceMarginal
        ? 'Expected OD use of a scheduled section is a service marginal: each value is the fitted OD demand times its cached probability of using that ride link. It is not an enumerated passenger path or an exact transfer-count distribution.'
        : `Estimated OD demand is separate from route-assignment flow. ${definition}. A service total is expected assigned demand or passenger-leg usage, not unique passengers when a journey uses multiple legs.`)
      : `${definition} Existing bundles without route-flow partitions support the OD destination view only.`;
  }

  function populateMeasurementOdControls() {
    const summary = viewer.summary.measurement_od_contributions || {};
    const locations = viewer.summary.measurement_locations || {};
    const onDemand = summary.available && summary.query_mode === 'local_service';
    const panel = $('measurement-od'); panel.classList.remove('hidden');
    $('measurement-od-direction').disabled = !summary.available;
    $('measurement-od-load').disabled = !summary.available;
    $('measurement-od-limit').disabled = !summary.available;
    $('measurement-query-service-label').classList.toggle('hidden', !onDemand);
    if (!summary.available) {
      $('measurement-od-unavailable').textContent = summary.message || 'OD-by-measurement contributions are unavailable in this bundle.';
      $('measurement-od-unavailable').classList.remove('hidden');
    } else {
      $('measurement-od-unavailable').classList.add('hidden');
    }
    const bins = canonicalBins((viewer.summary.od || {}).bins || []);
    options($('measurement-bin'), bins, 'Select departure interval');
    options($('measurement-origin'), (viewer.summary.od || {}).origins || [], 'Select origin');
    options($('measurement-destination'), (viewer.summary.od || {}).destinations || [], 'Select destination');
    if (bins.length) $('measurement-bin').value = bins[0];
    $('measurement-row-select').disabled = !locations.available;
    $('measurement-row-previous').disabled = !locations.available;
    $('measurement-row-next').disabled = !locations.available;
    $('map-measurement-mode').disabled = !locations.available;
    if (!locations.available) {
      const message = locations.message || 'Map-based measurement selection is unavailable in this bundle.';
      $('measurement-stop-label').textContent = message;
      $('map-click-status').textContent = message;
    }
    if (onDemand) {
      $('measurement-od-unavailable').textContent = 'Contributions are computed only for the selected measurement row by the local query service. Start the service documented in the viewer guide; no full OD-by-measurement expansion is stored in this bundle.';
      $('measurement-od-unavailable').classList.remove('hidden');
    }
    updateMeasurementOdDirection();
  }

  function updateMeasurementOdDirection() {
    const byMeasurement = $('measurement-od-direction').value === 'measurement';
    $('measurement-row-picker').classList.toggle('hidden', !byMeasurement);
    ['measurement-bin-label', 'measurement-origin-label', 'measurement-destination-label'].forEach((id) => $(id).classList.toggle('hidden', byMeasurement));
    $('measurement-contribution-previous').disabled = byMeasurement ? !viewer.selectedMeasurement : true;
    $('measurement-contribution-next').disabled = byMeasurement ? !viewer.selectedMeasurement : true;
  }

  function setMapClickMode(mode) {
    viewer.mapClickMode = mode;
    $('map-od-mode').classList.toggle('active', mode === 'od');
    $('map-measurement-mode').classList.toggle('active', mode === 'measurement');
    $('map-click-status').textContent = mode === 'measurement'
      ? 'Click a stop to inspect its measured rows.'
      : 'Click stops to choose an OD origin or destination.';
    if (mode === 'measurement') {
      const unmatched = viewer.measurementLocationManifest?.unmatched_stop_ids || [];
      if (unmatched.length) $('map-click-status').textContent += ` ${unmatched.length} measurement stop ID(s) have no exact network stop match; examples: ${unmatched.slice(0, 5).join(', ')}.`;
      const missingIds = Number(viewer.measurementLocationManifest?.missing_stop_rows) || 0;
      if (missingIds) $('map-click-status').textContent += ` ${missingIds} measurement row(s) have no stop ID.`;
    }
  }

  async function measurementComponents(measurementIndex) {
    const table = (viewer.summary.measurement_od_contributions || {}).component_table || {};
    if (Array.isArray(table.partitions)) {
      const descriptor = table.partitions.find((item) => measurementIndex >= Number(item.start_index) && measurementIndex < Number(item.start_index) + Number(item.row_count));
      if (!descriptor) return {};
      if (!String(descriptor.path || '').startsWith('partitions/measurement_od_contributions/components/') || String(descriptor.path || '').includes('..')) throw new Error('Measurement component partition path is unsafe.');
      const rows = await readDescriptorPage(descriptor, {measurement_index: String(measurementIndex)}, measurementIndex - Number(descriptor.start_index), 1, 'measurement-component');
      return rows[0] || {};
    }
    if (table.path) return (await readCsvFile(table.path, {measurement_index: String(measurementIndex)}, 1))[0] || {};
    return {};
  }

  async function renderSelectedMeasurementRow(row) {
    viewer.selectedMeasurement = row;
    viewer.measurementContributionPage = 0;
    $('measurement-row-select').value = String(row.measurement_index);
    $('measurement-row-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(row.measurement_index)}</dd><dt>Measurement type</dt><dd>${esc(row.measurement_type || '—')}</dd><dt>Line</dt><dd>${esc(row.line || '—')}</dd><dt>Direction</dt><dd>${esc(row.direction || '—')}</dd><dt>Stop</dt><dd>${esc(row.stop || '—')}</dd><dt>Observation time</dt><dd>${esc(row.observation_time || '—')}</dd><dt>Observation period</dt><dd>${esc(row.time_period || '—')}</dd><dt>Observed count</dt><dd>${display(row.observed)}</dd><dt>Modeled mean</dt><dd>${display(row.modeled)}</dd></dl>`;
    if (!(viewer.summary.measurement_od_contributions || {}).available) {
      $('measurement-od-detail').textContent = 'OD contributions are unavailable for this bundle; the measured row values above remain available.';
      renderMeasurementOdRows([]);
      return;
    }
    if ((viewer.summary.measurement_od_contributions || {}).query_mode === 'local_service') {
      $('measurement-od-detail').textContent = 'The local query service will calculate this row’s components and ranked OD contributors.';
      await refreshMeasurementOd();
      return;
    }
    try {
      const components = await measurementComponents(Number(row.measurement_index));
      const fixedAttribution = (viewer.summary.measurement_od_contributions || {}).fixed_positive_od_attribution_available === false
        ? `<p class="muted">${esc(viewer.summary.measurement_od_contributions.fixed_positive_od_attribution_message || 'Fixed-positive demand is aggregated in the mapping offset and cannot be assigned to individual OD pairs.')}</p>` : '';
      $('measurement-od-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(row.measurement_index)}</dd><dt>Observed count</dt><dd>${display(row.observed)}</dd><dt>Modeled mean</dt><dd>${display(components.modeled_mean ?? row.modeled)}</dd><dt>Fixed offset</dt><dd>${display(components.fixed_offset_contribution)}</dd><dt>Additive component</dt><dd>${display(components.additive_contribution)}</dd><dt>Mean-floor adjustment</dt><dd>${display(components.mean_floor_adjustment)}</dd></dl>${fixedAttribution}<p class="muted">OD contributors decompose the modeled mean; they are not a decomposition of the observed count.</p>`;
      await refreshMeasurementOd();
    } catch (error) {
      $('measurement-od-detail').textContent = `Measurement components unavailable: ${error.message}`;
    }
  }

  async function selectMeasurementStop(stopId) {
    stopMeasurementOdQuery('Select a measured row to inspect its OD decomposition.');
    viewer.selectedMeasurementStop = stopId;
    viewer.selectedMeasurement = null;
    viewer.measurementRowPage = 0;
    $('measurement-stop-label').textContent = `Measurements at stop ${stopId}`;
    $('measurement-row-select').innerHTML = '';
    $('measurement-row-select').disabled = true;
    $('measurement-row-detail').textContent = 'Loading measurements for this stop…';
    $('measurement-contribution-page').textContent = '';
    $('measurement-od-detail').textContent = 'No measurement selected.';
    renderMeasurementOdRows([]);
    try {
      const {rows, total} = await measurementsAtStop(stopId, 0, viewer.measurementRowsPerPage);
      if (!total) {
        viewer.measurementRows = [];
        $('measurement-row-detail').textContent = 'No measurements at this location.';
        $('measurement-stop-label').textContent = `No measurements at this location: ${stopId}`;
        $('measurement-row-page').textContent = '0 rows';
        $('map-click-status').textContent = `No measurements at this location: ${stopId}`;
        $('measurement-row-previous').disabled = true; $('measurement-row-next').disabled = true;
        return;
      }
      viewer.measurementRows = rows;
      $('measurement-row-select').disabled = false;
      $('measurement-row-select').innerHTML = rows.map((row) => {
        const details = [row.measurement_type, row.line, row.direction, row.observation_time ? `observed ${row.observation_time}` : '', row.time_period ? `period ${row.time_period}` : '', row.stop].filter(Boolean).join(' · ');
        return `<option value="${esc(row.measurement_index)}">${esc(row.measurement_index)} — ${esc(details)}</option>`;
      }).join('');
      $('measurement-row-page').textContent = `Rows ${viewer.measurementRowPage * viewer.measurementRowsPerPage + 1}–${viewer.measurementRowPage * viewer.measurementRowsPerPage + rows.length} of ${total}`;
      $('measurement-row-previous').disabled = viewer.measurementRowPage === 0;
      $('measurement-row-next').disabled = (viewer.measurementRowPage + 1) * viewer.measurementRowsPerPage >= total;
      await renderSelectedMeasurementRow(rows[0]);
      $('measurement-od').scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (error) {
      $('measurement-row-detail').textContent = `Could not read measurements for this stop: ${error.message}`;
      $('map-click-status').textContent = `Measurement lookup failed for ${stopId}: ${error.message}`;
    }
  }

  async function changeMeasurementRowPage(delta) {
    if (!viewer.selectedMeasurementStop) return;
    viewer.measurementRowPage = Math.max(0, viewer.measurementRowPage + delta);
    try {
      const {rows, total} = await measurementsAtStop(viewer.selectedMeasurementStop, viewer.measurementRowPage, viewer.measurementRowsPerPage);
      viewer.measurementRows = rows;
      $('measurement-row-select').innerHTML = rows.map((row) => {
        const details = [row.measurement_type, row.line, row.direction, row.observation_time ? `observed ${row.observation_time}` : '', row.time_period ? `period ${row.time_period}` : '', row.stop].filter(Boolean).join(' · ');
        return `<option value="${esc(row.measurement_index)}">${esc(row.measurement_index)} — ${esc(details)}</option>`;
      }).join('');
      $('measurement-row-page').textContent = `Rows ${viewer.measurementRowPage * viewer.measurementRowsPerPage + 1}–${viewer.measurementRowPage * viewer.measurementRowsPerPage + rows.length} of ${total}`;
      $('measurement-row-previous').disabled = viewer.measurementRowPage === 0;
      $('measurement-row-next').disabled = (viewer.measurementRowPage + 1) * viewer.measurementRowsPerPage >= total;
      if (rows.length) await renderSelectedMeasurementRow(rows[0]);
    } catch (error) { $('measurement-row-detail').textContent = `Could not read this measurement page: ${error.message}`; }
  }

  async function readCsvFile(name, filters = {}, limit = 5000) {
    const file = findFile(name); if (!file) throw new Error(`File is missing: ${name}`);
    return new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const collected = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
      worker.postMessage({file, filters, limit: Math.max(1, limit), chunkSize: 1024 * 1024});
    });
  }

  async function labelMeasurementOdRows(rows, direction) {
    if (!rows.length) return rows;
    if (direction === 'measurement') {
      const byBin = new Map();
      rows.forEach((row) => { const bin = row.departure_time_bin; if (!byBin.has(bin)) byBin.set(bin, []); byBin.get(bin).push(String(row.od_index)); });
      const odLabels = new Map();
      for (const [bin, ids] of byBin) {
        const partitionRows = await readPartition('od', bin, {od_index: [...new Set(ids)]}, new Set(ids).size);
        partitionRows.forEach((row) => odLabels.set(String(row.od_index), row));
      }
      const measurement = viewer.selectedMeasurement || {};
      return rows.map((row) => ({
        od_index: row.od_index,
        origin_stop_id: odLabels.get(String(row.od_index))?.origin_stop_id || '',
        destination_stop_id: odLabels.get(String(row.od_index))?.destination_stop_id || '',
        departure_time_bin: row.departure_time_bin,
        estimated_demand: odLabels.get(String(row.od_index))?.estimated_demand ?? '',
        measurement_index: row.measurement_index,
        measurement_type: measurement.measurement_type || '',
        line: measurement.line || '',
        direction: measurement.direction || '',
        stop: measurement.stop || '',
        observation_time: measurement.observation_time || '',
        time_period: measurement.time_period || '',
        observed: measurement.observed || '',
        modeled: measurement.modeled || '',
        linear_flow_contribution: row.linear_flow_contribution,
        modeled_mean_contribution: row.modeled_mean_contribution,
      }));
    }
    const indices = [...new Set(rows.map((row) => String(row.measurement_index)))];
    const measurements = await readCsvFile('predicted_measurements.csv', {row_index: indices}, indices.length);
    const byIndex = new Map(measurements.map((row) => [String(row.row_index), row]));
    return rows.map((row) => {
      const measurement = byIndex.get(String(row.measurement_index)) || {};
      return {
        measurement_index: row.measurement_index,
        measurement_type: measurement.measurement_type || '',
        line: measurement.line || '',
        direction: measurement.direction || '',
        stop: measurement.stop || '',
        observation_time: measurement.observation_time || '',
        time_period: measurement.time_period || '',
        observed: measurement.observed || '',
        modeled: measurement.modeled || '',
        departure_time_bin: row.departure_time_bin,
        operator_coefficient: row.operator_coefficient,
        linear_flow_contribution: row.linear_flow_contribution,
        modeled_mean_contribution: row.modeled_mean_contribution,
      };
    });
  }

  function renderMeasurementOdRows(rows) {
    const target = $('measurement-od-table');
    if (!rows.length) { target.querySelector('thead').innerHTML = ''; target.querySelector('tbody').innerHTML = '<tr><td class="muted">No OD contribution entries match this selection.</td></tr>'; return; }
    const headers = Object.keys(rows[0]);
    target.querySelector('thead').innerHTML = `<tr>${headers.map((header) => `<th>${esc(header)}</th>`).join('')}</tr>`;
    target.querySelector('tbody').innerHTML = rows.slice(0, 500).map((row) => `<tr>${headers.map((header) => `<td>${esc(row[header])}</td>`).join('')}</tr>`).join('');
  }

  function formatElapsed(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return minutes ? `${minutes}m ${String(remainder).padStart(2, '0')}s` : `${remainder}s`;
  }

  function stopMeasurementOdQuery(message) {
    viewer.measurementQueryId += 1;
    if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
    viewer.measurementQueryTimer = null;
    const status = $('measurement-od-status');
    status.classList.remove('is-loading');
    status.classList.remove('bad', 'ok');
    status.setAttribute('aria-busy', 'false');
    const button = $('measurement-od-load');
    button.disabled = false;
    button.textContent = 'Load contributions';
    if (message) status.textContent = message;
  }

  async function refreshMeasurementOd() {
    const summary = viewer.summary.measurement_od_contributions || {};
    if (!summary.available) return;
    const direction = $('measurement-od-direction').value;
    const limit = Math.max(1, Math.min(500, Number($('measurement-od-limit').value) || 50));
    const localService = summary.available && summary.query_mode === 'local_service';
    if (localService && direction === 'od' && summary.reverse_query_available === false) {
      stopMeasurementOdQuery();
      $('measurement-od-status').textContent = summary.reverse_query_message || 'Reverse OD-to-measurement contributions are unavailable for this fit.';
      $('measurement-od-status').classList.add('bad');
      $('measurement-od-detail').textContent = summary.reverse_query_message || 'Reverse OD-to-measurement contributions are unavailable for this fit.';
      renderMeasurementOdRows([]);
      return;
    }
    const onDemand = direction === 'measurement' && summary.query_mode === 'local_service';
    const requestId = ++viewer.measurementQueryId;
    if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
    const status = $('measurement-od-status');
    const button = $('measurement-od-load');
    const startedAt = Date.now();
    const measurementIndex = direction === 'measurement' ? Number(viewer.selectedMeasurement?.measurement_index) : null;
    const operation = onDemand
      ? `Calculating OD contributions for measurement row ${measurementIndex} from the local mapping`
      : localService ? 'Querying the selected OD cell in the local mapping service'
        : direction === 'measurement' ? 'Loading this measurement’s sparse contributors into the local cache' : 'Reading indexed sparse contribution partitions';
    const updateLoadingStatus = () => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      const progressHint = elapsed >= 15
        ? 'Still processing; the selected mapping data can take a few minutes to scan. No percentage estimate is available.'
        : 'Please wait while the selected mapping data is processed.';
      status.textContent = `${operation}… Elapsed ${formatElapsed(elapsed)}. ${progressHint}`;
    };
    status.classList.remove('bad', 'ok');
    status.classList.add('is-loading');
    status.setAttribute('aria-busy', 'true');
    button.disabled = true;
    button.textContent = 'Calculating…';
    updateLoadingStatus();
    viewer.measurementQueryTimer = setInterval(updateLoadingStatus, 10000);
    try {
      let rows; let selectedOd = null; let total = 0; let queryResult = null;
      if (direction === 'measurement') {
        if (!viewer.selectedMeasurement) throw new Error('Select a measurement row from the map first.');
        queryResult = onDemand
          ? await queryLocalMeasurementPage(measurementIndex, viewer.measurementContributionPage, limit)
          : await measurementContributionPage(measurementIndex, viewer.measurementContributionPage, limit);
        rows = queryResult.rows; total = queryResult.total;
        if (onDemand) {
          const components = queryResult.components || {};
          const fixedAttribution = summary.fixed_positive_od_attribution_available === false
            ? `<p class="muted">${esc(summary.fixed_positive_od_attribution_message || 'Fixed-positive demand is aggregated in the mapping offset and cannot be assigned to individual OD pairs.')}</p>` : '';
          $('measurement-od-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(measurementIndex)}</dd><dt>Observed count</dt><dd>${display(viewer.selectedMeasurement.observed)}</dd><dt>Modeled mean</dt><dd>${display(components.modeled_mean ?? viewer.selectedMeasurement.modeled)}</dd><dt>Fixed offset</dt><dd>${display(components.fixed_offset_contribution)}</dd><dt>Additive component</dt><dd>${display(components.additive_contribution)}</dd><dt>OD component</dt><dd>${display(components.od_contribution)}</dd><dt>Mean-floor adjustment</dt><dd>${display(components.mean_floor_adjustment)}</dd></dl>${fixedAttribution}<p class="muted">OD contributors decompose the modeled mean; they are not a decomposition of the observed count.</p>`;
        }
      } else {
        const bin = $('measurement-bin').value; const origin = $('measurement-origin').value; const destination = $('measurement-destination').value;
        if (!bin || !origin || !destination) throw new Error('Choose a departure interval, origin, and destination.');
        const odRows = await readPartition('od', bin, {origin_stop_id: origin, destination_stop_id: destination}, 100);
        selectedOd = odRows.find((row) => row.departure_time_bin === bin);
        if (!selectedOd) { renderMeasurementOdRows([]); $('measurement-od-status').textContent = 'No fitted OD cell matches that selection.'; return; }
        if (localService) {
          queryResult = await queryLocalOdPage(Number(selectedOd.od_index), limit);
          rows = queryResult.rows; total = queryResult.total;
        } else {
          rows = await readPartition('measurement_od_contributions', bin, {od_index: String(selectedOd.od_index)}, limit, {rankBy: 'modeled_mean_contribution', rankAbs: true});
        }
      }
      if (requestId !== viewer.measurementQueryId) return;
      rows = await labelMeasurementOdRows(rows, direction);
      if (requestId !== viewer.measurementQueryId) return;
      renderMeasurementOdRows(rows);
      if (direction === 'measurement') {
        const pageCount = Math.max(1, Math.ceil(total / limit));
        $('measurement-contribution-page').textContent = `Page ${viewer.measurementContributionPage + 1} of ${pageCount} · ${total} OD contributors, ranked by absolute modeled-mean contribution`;
        $('measurement-contribution-previous').disabled = viewer.measurementContributionPage === 0;
        $('measurement-contribution-next').disabled = (viewer.measurementContributionPage + 1) * limit >= total;
      } else {
        $('measurement-od-detail').innerHTML = `<dl><dt>OD pair</dt><dd>${esc(selectedOd.origin_stop_id)} → ${esc(selectedOd.destination_stop_id)} (${esc(selectedOd.departure_time_bin)})</dd><dt>Estimated demand</dt><dd>${display(selectedOd.estimated_demand)}</dd><dt>Measurement rows shown</dt><dd>${esc(rows.length)}${localService ? ` of ${total}` : ''} (ranked by absolute modeled contribution)</dd></dl>`;
      }
      $('measurement-od-status').textContent = direction === 'measurement'
        ? `Loaded ${rows.length} of ${total} OD contributors in ${formatElapsed(Math.floor((Date.now() - startedAt) / 1000))}. Signed modeled-mean and linear-flow contributions are preserved.`
        : `Loaded ${rows.length} bounded contribution rows in ${formatElapsed(Math.floor((Date.now() - startedAt) / 1000))}.`;
      status.classList.add('ok');
    } catch (error) {
      if (requestId === viewer.measurementQueryId) {
        const elapsed = formatElapsed(Math.floor((Date.now() - startedAt) / 1000));
        status.textContent = `Could not load OD contributions after ${elapsed}: ${error.message}`;
        status.classList.add('bad');
      }
    } finally {
      if (requestId === viewer.measurementQueryId) {
        if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
        viewer.measurementQueryTimer = null;
        status.classList.remove('is-loading');
        status.setAttribute('aria-busy', 'false');
        button.disabled = false;
        button.textContent = 'Load contributions';
      }
    }
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
    try {
      const locationManifest = await measurementLocationIndex();
      const unmatched = locationManifest?.unmatched_stop_ids || [];
      const missingIds = Number(locationManifest?.missing_stop_rows) || 0;
      if (unmatched.length || missingIds) {
        const messages = [];
        if (unmatched.length) messages.push(`${unmatched.length} measurement stop ID(s) do not exactly match network/stops.csv and cannot be selected on the map. Examples: ${unmatched.slice(0, 20).join(', ')}.`);
        if (missingIds) messages.push(`${missingIds} measurement row(s) have a blank stop ID and are not selectable from the map.`);
        $('measurement-unmatched-notice').textContent = messages.join(' ');
        $('measurement-unmatched-notice').classList.remove('hidden');
      }
    } catch (error) {
      $('measurement-unmatched-notice').textContent = `Measurement-location index could not be validated: ${error.message}`;
      $('measurement-unmatched-notice').classList.remove('hidden');
      $('map-measurement-mode').disabled = true;
    }
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
    const indexedStops = viewer.measurementLocationManifest?.stops || {};
    viewer.stops.forEach((stop) => {
      const lat = num(stop.lat); const lon = num(stop.lon); if (lat === null || lon === null) return;
      const hasMeasurements = Object.prototype.hasOwnProperty.call(indexedStops, stop.stop_id);
      const rowCount = hasMeasurements ? Number(indexedStops[stop.stop_id].row_count) || 0 : 0;
      const marker = L.circleMarker([lat, lon], {radius: hasMeasurements ? 6 : 4, color: hasMeasurements ? '#087f8c' : '#235aa6', fillColor: hasMeasurements ? '#40c4cf' : '#235aa6', fillOpacity: .75});
      marker.bindTooltip(`${esc(stop.stop_id)}: ${esc(stop.name || '')}${hasMeasurements ? ` · ${rowCount} measurement row(s)` : ''}`)
        .on('click', () => viewer.mapClickMode === 'measurement' ? selectMeasurementStop(stop.stop_id) : chooseOdStop(stop.stop_id))
        .addTo(viewer.markerLayer);
    });
    await renderLineGeometry();
  }

  async function renderLineGeometry() {
    const geometry = viewer.summary.line_geometry || viewer.summary.route_sections || {};
    if (!geometry.available || !geometry.geometry_path || !viewer.map || !window.L) return;
    const file = findFile(geometry.geometry_path);
    if (!file) return;
    try {
      const geojson = JSON.parse(await file.text());
      L.geoJSON(geojson, {
        style: {color: '#6a4fb3', weight: 4, opacity: .75},
        interactive: false,
      }).addTo(viewer.map);
    } catch (error) {
      setFlowStatus(`Line geometry unavailable: ${error.message}`, '');
    }
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
          measurement_od_contributions: {available: false, message: 'Legacy bundle has no OD-by-measurement contribution dataset.'},
        };
      }
      if (!viewer.summary.measurement_od_contributions) viewer.summary.measurement_od_contributions = {available: false, message: 'This bundle predates OD-by-measurement contributions.'};
      viewer.cacheIdentity = `${viewer.manifest.model_fingerprint || ''}|${viewer.manifest.specification_fingerprint || ''}|${viewer.manifest.report_fingerprint || ''}|${JSON.stringify(viewer.manifest.provenance || {})}`;
      $('summary').classList.remove('hidden'); $('diagnostics').classList.remove('hidden'); $('tables').classList.remove('hidden'); $('flows').classList.remove('hidden');
      const counts = viewer.manifest.row_counts || {}; $('metrics').innerHTML = [metric('Bundle type', viewer.manifest.bundle_type), metric('Schema', viewer.manifest.schema_version), metric('OD rows', counts['full_od.csv'] ?? 'unknown'), metric('Measurements', counts['predicted_measurements.csv'] ?? 'unknown'), metric('OD partitions', viewer.summary.od?.partitions?.length ?? 'legacy'), metric('Route flows', viewer.summary.route_flows?.available ? 'available' : 'absent'), metric('OD contribution map', viewer.summary.measurement_od_contributions?.available ? 'available' : 'absent'), metric('Identifiability', viewer.manifest.identifiability?.available ? 'available' : 'not available'), metric('Network files', Object.keys(viewer.manifest.files || {}).filter((key) => key.startsWith('network/')).length)].join('');
      const modelLikelihood = viewer.model.likelihood || viewer.model.specification?.likelihood?.family; $('model').innerHTML = renderKeyValue({fingerprint: viewer.manifest.model_fingerprint, likelihood: modelLikelihood}); $('provenance').innerHTML = renderKeyValue(viewer.manifest.provenance || viewer.manifest.fit_provenance); const adequacy = viewer.report.adequacy || {}; $('diagnostic-metrics').innerHTML = [metric('Observed total', adequacy.observed_total), metric('Modeled total', adequacy.modeled_total), metric('MAE', adequacy.mae), metric('RMSE', adequacy.rmse), metric('Report fingerprint', viewer.report.report_fingerprint, 'fingerprint')].join('');
      $('table-content').innerHTML = '<p class="muted">Large tables are available through the partitioned explorer. No complete CSV is parsed at startup.</p>';
      const executive = findFile('executive_summary.md'); $('executive').innerHTML = executive ? `<pre>${esc(await executive.text())}</pre>` : '<p>No executive summary supplied.</p>'; populateFlowSelectors(); populateMeasurementOdControls(); await renderNetwork(); await refreshFlows();
      if (!viewer.summary.od) { $('interactive-notice').textContent = 'This is a legacy bundle without partition metadata. The validated tables remain available; route-flow and lazy time-slot exploration require a refreshed export.'; $('interactive-notice').classList.remove('hidden'); }
      if (problems.length) setStatus(`Bundle loaded with startup validation warnings:\n- ${problems.join('\n- ')}`, 'bad'); else setStatus('Bundle manifest and summary loaded. Partition checksums are verified when selected.', 'ok');
    } catch (error) { $('summary').classList.add('hidden'); $('diagnostics').classList.add('hidden'); $('flows').classList.add('hidden'); setStatus(`Could not load bundle: ${error.message}`, 'bad'); }
  }
  function metric(label, value, extra = '') { return `<div class="metric"><label>${esc(label)}</label><strong class="${extra}">${esc(value)}</strong></div>`; }
  function renderKeyValue(value) { if (!value || typeof value !== 'object') return `<p>${esc(value)}</p>`; return `<div class="fingerprint">${Object.entries(value).map(([key, item]) => `<div><b>${esc(key)}:</b> ${esc(typeof item === 'object' ? JSON.stringify(item) : item)}</div>`).join('')}</div>`; }
  $('folder').addEventListener('change', (event) => load([...event.target.files])); $('flow-refresh').addEventListener('click', refreshFlows); $('flow-clear-cache').addEventListener('click', clearCache); ['flow-bin', 'flow-origin', 'flow-destination', 'flow-line', 'flow-trip', 'flow-transfers', 'flow-kind'].forEach((id) => { if ($(id)) $(id).addEventListener('change', refreshFlows); });
  $('map-od-mode')?.addEventListener('click', () => setMapClickMode('od')); $('map-measurement-mode')?.addEventListener('click', () => setMapClickMode('measurement'));
  $('measurement-od-direction')?.addEventListener('change', updateMeasurementOdDirection); $('measurement-od-load')?.addEventListener('click', refreshMeasurementOd);
  $('measurement-row-select')?.addEventListener('change', () => {
    const row = viewer.measurementRows.find((item) => String(item.measurement_index) === $('measurement-row-select').value);
    if (row) renderSelectedMeasurementRow(row);
  });
  $('measurement-row-previous')?.addEventListener('click', () => changeMeasurementRowPage(-1)); $('measurement-row-next')?.addEventListener('click', () => changeMeasurementRowPage(1));
  $('measurement-contribution-previous')?.addEventListener('click', () => { viewer.measurementContributionPage = Math.max(0, viewer.measurementContributionPage - 1); refreshMeasurementOd(); });
  $('measurement-contribution-next')?.addEventListener('click', () => { viewer.measurementContributionPage += 1; refreshMeasurementOd(); });
  $('measurement-od-limit')?.addEventListener('change', () => { viewer.measurementContributionPage = 0; if (viewer.selectedMeasurement) refreshMeasurementOd(); });
  ['measurement-bin', 'measurement-origin', 'measurement-destination'].forEach((id) => $(id)?.addEventListener('change', () => { if ($('measurement-od-direction').value === 'od') refreshMeasurementOd(); }));
})();
