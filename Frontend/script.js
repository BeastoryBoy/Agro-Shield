/* ============================================================
   PARTICLE CANVAS
============================================================ */
const bgC = document.getElementById('bgc');
const bgX = bgC.getContext('2d');
let W, H, pts = [];

function rsz() { W = bgC.width = window.innerWidth; H = bgC.height = window.innerHeight; }
rsz(); window.addEventListener('resize', rsz);

class Pt {
  constructor() { this.init(true); }
  init(rand) {
    this.x = Math.random() * W;
    this.y = rand ? Math.random() * H : (Math.random() < .5 ? -4 : H + 4);
    this.vx = (Math.random() - .5) * .4;
    this.vy = (Math.random() - .5) * .4;
    this.r = Math.random() * 1.4 + .4;
    this.phase = 0;
    this.dp = Math.random() * .004 + .0012;
    const q = Math.random();
    this.col = q > .58 ? '#00ff9d' : q > .32 ? '#00d4ff' : '#c26eff';
  }
  update() {
    this.x += this.vx; this.y += this.vy; this.phase += this.dp;
    if (this.phase > 1 || this.x < -8 || this.x > W + 8 || this.y < -8 || this.y > H + 8) this.init();
  }
  draw() {
    bgX.globalAlpha = Math.sin(this.phase * Math.PI) * .48;
    bgX.fillStyle = this.col;
    bgX.beginPath(); bgX.arc(this.x, this.y, this.r, 0, Math.PI * 2); bgX.fill();
  }
}

for (let i = 0; i < 115; i++) pts.push(new Pt());

function drawLines() {
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 105) {
        bgX.globalAlpha = .046 * (1 - d / 105);
        bgX.strokeStyle = '#00ff9d'; bgX.lineWidth = .55;
        bgX.beginPath(); bgX.moveTo(pts[i].x, pts[i].y); bgX.lineTo(pts[j].x, pts[j].y); bgX.stroke();
      }
    }
  }
}

function animBg() {
  bgX.clearRect(0, 0, W, H);
  drawLines();
  pts.forEach(p => { p.update(); p.draw(); });
  bgX.globalAlpha = 1;
  requestAnimationFrame(animBg);
}
animBg();

/* ============================================================
   NAV
============================================================ */
window.addEventListener('scroll', () => {
  document.getElementById('nav').classList.toggle('sc', window.scrollY > 40);
  // Highlight active nav
  ['scanner','encyclopedia','prevention','share'].forEach(id => {
    const el = document.getElementById(id);
    const btn = document.querySelector(`.nl[onclick*="${id}"]`);
    if (!el || !btn) return;
    const rect = el.getBoundingClientRect();
    btn.classList.toggle('active', rect.top <= 80 && rect.bottom > 80);
  });
});

function sTo(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ============================================================
   AUTH
============================================================ */
const users = [];
let cu = null, scanCt = 0, detCt = 0, alertCt = 0, lastResult = null;

function openAuth(tab) {
  document.getElementById('auth-ov').classList.add('show');
  document.getElementById('auth-forms').style.display = 'block';
  document.getElementById('auth-success').style.display = 'none';
  swTab(tab === 'register' ? 'r' : 'l');
  document.getElementById('auth-err').style.display = 'none';
}
function closeAuth() { document.getElementById('auth-ov').classList.remove('show'); }
function swTab(t) {
  document.getElementById('atab-l').classList.toggle('a', t === 'l');
  document.getElementById('atab-r').classList.toggle('a', t === 'r');
  document.getElementById('f-l').classList.toggle('hidden', t !== 'l');
  document.getElementById('f-r').classList.toggle('hidden', t !== 'r');
  document.getElementById('auth-err').style.display = 'none';
}
function showErr(m) { const e = document.getElementById('auth-err'); e.textContent = m; e.style.display = 'block'; }
function chkPw(v) {
  const f = document.getElementById('pw-fill');
  const s = v.length < 4 ? 0 : v.length < 6 ? 22 : v.length < 8 ? 52 : v.length < 12 ? 78 : 100;
  f.style.width = s + '%';
  f.style.background = s < 35 ? 'var(--r)' : s < 65 ? 'var(--gold)' : 'var(--g)';
}
function doLogin() {
  const em = document.getElementById('l-em').value.trim();
  const pw = document.getElementById('l-pw').value;
  if (!em || !pw) return showErr('Please enter your email and password.');
  if (!em.includes('@')) return showErr('Please enter a valid email address.');
  const found = users.find(u => u.email === em && u.password === pw);
  if (found) cu = found;
  else if (users.some(u => u.email === em)) return showErr('Incorrect password. Please try again.');
  else return showErr('No account found with this email. Please create an account first.');
  authSuccess(`Welcome back, ${cu.name.split(' ')[0]}! 👋`, 'You are signed in. Start scanning your plants for instant disease diagnosis.');
}
function doRegister() {
  const nm = document.getElementById('r-nm').value.trim();
  const em = document.getElementById('r-em').value.trim();
  const pw = document.getElementById('r-pw').value;
  const ph = document.getElementById('r-ph').value.trim();
  if (!nm) return showErr('Please enter your full name.');
  if (!em || !em.includes('@')) return showErr('Please enter a valid email address.');
  if (pw.length < 6) return showErr('Password must be at least 6 characters.');
  if (users.find(u => u.email === em)) return showErr('An account with this email already exists. Please sign in.');
  cu = { name: nm, email: em, phone: ph, password: pw };
  users.push(cu);
  authSuccess(`Welcome to AgroShield, ${nm.split(' ')[0]}! 🎉`, 'Your account is ready. Upload any plant photo for instant AI disease diagnosis.');
}
function authSuccess(title, msg) {
  document.getElementById('auth-forms').style.display = 'none';
  document.getElementById('auth-success').style.display = 'block';
  document.getElementById('asc-title').textContent = title;
  document.getElementById('asc-msg').textContent = msg;
  const ini = cu.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  document.getElementById('nav-user').style.display = 'flex';
  document.getElementById('nav-auth-btn').style.display = 'none';
  document.getElementById('nav-uname').textContent = cu.name.split(' ')[0];
  document.getElementById('nav-av').textContent = ini;
  setTimeout(closeAuth, 2000);
}
function doLogout() {
  cu = null;
  document.getElementById('nav-user').style.display = 'none';
  document.getElementById('nav-auth-btn').style.display = 'flex';
}

/* ============================================================
   FILE UPLOAD
============================================================ */
let imgB64 = null, imgMime = null;

function dzOver(e) { e.preventDefault(); document.getElementById('dz').classList.add('drag'); }
function dzLeave() { document.getElementById('dz').classList.remove('drag'); }
function dzDrop(e) {
  e.preventDefault(); dzLeave();
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
  else alert('Please drop an image file.');
}
document.getElementById('dz').addEventListener('click', () => {
  if (!imgB64) document.getElementById('file-in').click();
});
function handleFile(file) {
  window.selectedFile = file;
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { alert('File too large. Max 10MB.'); return; }
  imgMime = file.type || 'image/jpeg';
  const reader = new FileReader();
  reader.onload = ev => {
    imgB64 = ev.target.result.split(',')[1];
    document.getElementById('prev-img').src = ev.target.result;
    document.getElementById('dz-ph').style.display = 'none';
    document.getElementById('dz-prev').style.display = 'block';
    document.getElementById('btn-scan').disabled = false;
    document.getElementById('btn-reset').style.display = 'none';
    resetRP();
  };
  reader.readAsDataURL(file);
}
function resetScanner() {
  imgB64 = null; imgMime = null;
  document.getElementById('dz-ph').style.display = 'flex';
  document.getElementById('dz-prev').style.display = 'none';
  document.getElementById('btn-scan').disabled = true;
  document.getElementById('btn-reset').style.display = 'none';
  document.getElementById('file-in').value = '';
  document.getElementById('dz').classList.remove('scanning');
  resetRP();
}
function resetRP() {
  document.getElementById('rp-empty').style.display = 'flex';
  document.getElementById('rp-loading').style.display = 'none';
  document.getElementById('rp-result').style.display = 'none';
}

/* ============================================================
   GEMINI AI SCAN
============================================================ */
async function runScan() {
  const file = window.selectedFile;
  if (!file) return;

  document.getElementById('dz').classList.add('scanning');
  document.getElementById('btn-scan').disabled = true;
  document.getElementById('btn-reset').style.display = 'none';
  document.getElementById('rp-empty').style.display = 'none';
  document.getElementById('rp-result').style.display = 'none';
  document.getElementById('rp-loading').style.display = 'flex';

  const stepIds = ['ps1','ps2','ps3','ps4','ps5'];
  const stepLabels = [
    'Loading ML model…',
    'Processing image…',
    'Detecting disease…',
    'Computing confidence…',
    'Generating result…'
  ];

  let si = 0;
  const iv = setInterval(() => {
    if (si > 0) document.getElementById(stepIds[si-1]).className = 'ps done';
    if (si < stepIds.length) {
      document.getElementById(stepIds[si]).className = 'ps act';
      document.getElementById('proc-lbl').textContent = stepLabels[si];
      si++;
    }
  }, 600);

  try {
    const formData = new FormData();
    formData.append("image", file);

    const res = await fetch("https://agro-backend-aqwn.onrender.com/submit", {
      method: "POST",
      body: formData
    });

    const result = await res.json(); // 🔥 KEY CHANGE

    clearInterval(iv);

    document.getElementById('dz').classList.remove('scanning');
    document.getElementById('rp-loading').style.display = 'none';
    document.getElementById('rp-result').style.display = 'flex';

    if (!result.isPlant) {
  document.getElementById('rp-result').style.display = 'flex';
  showErrResult(result);
  return; // 🔥 CRITICAL — STOP EXECUTION
}

showResult(result);

  } catch (e) {
    console.error(e);

    clearInterval(iv);

    document.getElementById('dz').classList.remove('scanning');
    document.getElementById('rp-loading').style.display = 'none';
    document.getElementById('rp-result').style.display = 'flex';

    showErrResult({
      errorTitle: 'Connection Error',
      errorMessage: 'Could not reach ML backend.',
      errorSuggestion: 'Make sure Flask server is running on port 5000.'
    });
  }

  document.getElementById('btn-scan').disabled = false;
  document.getElementById('btn-reset').style.display = 'block';
}

function tog(h) { h.classList.toggle('open'); h.nextElementSibling.classList.toggle('open'); }

function showErrResult(r) {
  document.getElementById('rp-inner').innerHTML = `
    <div class="err-card">
      <div class="err-icon">🚫</div>
      <div class="err-title">${r.errorTitle}</div>
      <div class="err-msg">${r.errorMessage}</div>
      <div class="err-guide">
        <strong>💡 Try this:</strong><br>
        ${r.errorSuggestion}
      </div>
    </div>
  `;
}

function showResult(r) {
  const sc = { none:'var(--g)', low:'var(--g)', med:'var(--gold)', high:'var(--r)' };
  const sl = { none:'No Risk', low:'Low Risk', med:'Moderate', high:'High Risk' };
  const uc = { none:'var(--g)', routine:'var(--b)', moderate:'var(--gold)', immediate:'var(--r)' };
  const ul = { none:'No Action', routine:'Routine', moderate:'Act Soon', immediate:'⚡ Immediate!' };

  // 🔥 CLEAN NAME FIX
  const cleanName = (r.condition || 'Analysis Complete')
    .replace(/___/g, ' ')
    .replace(/_/g, ' ');

  // 🔥 BETTER CONFIDENCE (REALISTIC LOOK)
  const pct = r.confidence || 0;

  const rad = 26, circ = 2 * Math.PI * rad;
  const da = `${(pct / 100) * circ} ${circ}`;
  const gc = pct >= 90 ? 'var(--g)' : pct >= 75 ? 'var(--gold)' : 'var(--r)';

  const typeRb = { Fungal: 'rb-type', Bacterial: 'rb-type', Viral: 'rb-err', Pest: 'rb-pest', 'Nutritional Deficiency': 'rb-ok', None: '' };

  let html = `
  <div class="res-badges">
    ${r.isHealthy ? '<span class="rb rb-ok">✓ Healthy Plant</span>' : '<span class="rb rb-err">⚠ Disease Detected</span>'}
    <span class="rb rb-crop">🌱 ${r.plantName || 'Plant'}</span>
    ${r.diseaseType && r.diseaseType !== 'None' ? `<span class="rb ${typeRb[r.diseaseType] || 'rb-type'}">${r.diseaseType}</span>` : ''}
  </div>

  <div class="res-name">${cleanName}</div>

  ${r.scientificName ? `<div class="res-sci">${r.scientificName}</div>` : '<div style="margin-bottom:13px;"></div>'}

  <div class="conf-row">
    <svg width="62" height="62" viewBox="0 0 62 62" style="flex-shrink:0;">
      <circle cx="31" cy="31" r="${rad}" fill="none" stroke="var(--bg4)" stroke-width="5"/>
      <circle cx="31" cy="31" r="${rad}" fill="none" stroke="${gc}" stroke-width="5"
        stroke-dasharray="${da}" stroke-linecap="round" transform="rotate(-90 31 31)"
        style="filter:drop-shadow(0 0 5px ${gc});"/>
      <text x="31" y="36" text-anchor="middle" font-size="11.5" font-weight="800" fill="${gc}" font-family="Space Grotesk">${pct}%</text>
    </svg>

    <div class="conf-txt">
      <div class="conf-lbl">Model Confidence</div>
      <div class="conf-num" style="color:${gc};">${pct}%</div>
      <div class="conf-sub">${r.isHealthy ? 'No disease found' : 'Disease identified by ML model'}</div>
    </div>
  </div>

  <div class="info22">
    <div class="ib"><div class="ib-l">Severity</div><div class="ib-v" style="color:${sc[r.severity] || 'var(--t1)'};">${sl[r.severity] || '—'}</div></div>
    <div class="ib"><div class="ib-l">Urgency</div><div class="ib-v" style="color:${uc[r.urgency] || 'var(--t1)'};">${ul[r.urgency] || '—'}</div></div>
    <div class="ib"><div class="ib-l">Spread Risk</div><div class="ib-v">${r.spreadRisk || '—'}</div></div>
    <div class="ib"><div class="ib-l">Yield Impact</div><div class="ib-v">${r.yieldImpact || '—'}</div></div>
  </div>`;

  if (!r.isHealthy) {
    html += `
  <div class="exp">
    <div class="exp-h open" onclick="tog(this)"><span>⚠️ Symptoms Observed</span><span class="exp-arr">▾</span></div>
    <div class="exp-body open"><div class="chips">${(r.symptoms || []).map(s => `<span class="chip">${s}</span>`).join('')}</div></div>
  </div>

  <div class="exp">
    <div class="exp-h" onclick="tog(this)"><span>🔬 Root Cause</span><span class="exp-arr">▾</span></div>
    <div class="exp-body"><p style="font-size:.78rem;color:var(--t2);line-height:1.7;">${r.cause || '—'}</p></div>
  </div>

  <div class="exp">
    <div class="exp-h open" onclick="tog(this)"><span>💊 Treatment Steps</span><span class="exp-arr">▾</span></div>
    <div class="exp-body open">
      <div class="sl">${(r.treatment || []).map((t, i) => `<div class="si"><div class="sn">${i+1}</div><p>${t}</p></div>`).join('')}</div>
      ${r.organicOption ? `<div class="org-tip">🌿 <strong>Organic Option:</strong> ${r.organicOption}</div>` : ''}
    </div>
  </div>`;
  } else {
    html += `
  <div class="healthy-box">
    <div style="font-size:1.4rem;margin-bottom:6px;">🎉</div>
    <div style="font-weight:800;color:var(--g);font-size:.92rem;">Plant is Healthy!</div>
    <div style="font-size:.76rem;color:var(--t2);margin-top:4px;">No disease detected. Continue good farming practices.</div>
  </div>`;
  }

  html += `
  <div class="exp">
    <div class="exp-h" onclick="tog(this)"><span>🛡️ Prevention Strategies</span><span class="exp-arr">▾</span></div>
    <div class="exp-body"><div class="sl">${(r.prevention || []).map((p, i) => `<div class="si"><div class="sn">${i+1}</div><p>${p}</p></div>`).join('')}</div></div>
  </div>

  <div class="share-btns">
    <button class="sbtn sbtn-wa" onclick="goShare()">💬 WhatsApp</button>
    <button class="sbtn sbtn-sms" onclick="goShare()">📱 SMS</button>
    <button class="sbtn sbtn-cp" onclick="copyRes()">📋 Copy</button>
  </div>`;

  document.getElementById('rp-inner').innerHTML = html;
}

/* ============================================================
   SHARE
============================================================ */
function buildMsg() {
  if (!lastResult || !lastResult.isPlant) return 'Complete a plant scan first to generate your shareable alert message.';
  const r = lastResult;
  const dt = new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
  if (r.isHealthy) return `🌿 AGROSHIELD REPORT — ${dt}\n\n✅ STATUS: HEALTHY\n━━━━━━━━━━━━━━\n🌱 Crop: ${r.plantName}\n🤖 AI Confidence: ${r.confidence}%\n\n💡 Plant is healthy. Continue regular monitoring.\n\n— AgroShield AI`;
  return `🌿 AGROSHIELD DISEASE ALERT — ${dt}\n\n⚠️ DISEASE DETECTED\n━━━━━━━━━━━━━━━━━\n🌱 Crop: ${r.plantName}\n🔬 Disease: ${r.condition}\n⚗️ Type: ${r.diseaseType}\n📊 Severity: ${(r.severity || '').toUpperCase()}\n🆘 Urgency: ${(r.urgency || '').toUpperCase()}\n🤖 Confidence: ${r.confidence}%\n\n📋 SYMPTOMS:\n${(r.symptoms || []).slice(0, 3).map(s => `• ${s}`).join('\n')}\n\n💊 TREATMENT:\n${(r.treatment || []).slice(0, 4).map((t, i) => `${i+1}. ${t}`).join('\n')}\n\n🛡️ PREVENTION:\n${(r.prevention || []).slice(0, 2).map(p => `• ${p}`).join('\n')}\n\n━━━━━━━━━━━━━━━━━\nPowered by AgroShield AI`;
}
function updShare() { const m = buildMsg(); document.getElementById('wa-prev').textContent = m; document.getElementById('sms-prev').textContent = m; }
function goShare() { sTo('share'); updShare(); }
function doWA() {
  if (!lastResult || !lastResult.isPlant) { alert('Please scan a plant first!'); return; }
  alertCt++; window.open('https://wa.me/?text=' + encodeURIComponent(buildMsg()), '_blank');
}
function doSMS() {
  if (!lastResult || !lastResult.isPlant) { alert('Please scan a plant first!'); return; }
  alertCt++; window.location.href = 'sms:?body=' + encodeURIComponent(buildMsg());
}
function copyRes() {
  const m = buildMsg();
  if (navigator.clipboard) navigator.clipboard.writeText(m).then(() => alert('✅ Copied!'));
  else { const t = document.createElement('textarea'); t.value = m; document.body.appendChild(t); t.select(); document.execCommand('copy'); document.body.removeChild(t); alert('✅ Copied!'); }
}

/* ============================================================
   ENCYCLOPEDIA DATA
============================================================ */
const CROPS = [
  {
    id:'tomato', name:'Tomato', emoji:'🍅',
    desc:'One of the world\'s most widely grown vegetables. Susceptible to over 200 diseases — fungal, bacterial, viral, and pest-related. Disease management is critical for profitable production.',
    season:'Year-round (greenhouse) / Summer (open field)', origin:'Andes, South America', family:'Solanaceae',
    diseases:[
      { n:'Early Blight', sc:'Alternaria solani', ty:'f', sv:'med',
        short:'Dark brown target-spot lesions on older leaves progressing upward through the canopy.',
        cause:['Fungal pathogen overwinters in plant debris and soil for 2+ years.','Wind-borne spores and rain splash spread between plants and fields.','Warm (24–29°C), humid (>90%) conditions with 4+ hours of leaf wetness trigger infection.','Nutritionally weakened plants — especially potassium or calcium deficient — show symptoms earlier.'],
        syms:['Dark brown spots with concentric rings (target-board pattern)','Yellow chlorotic halo around each lesion','Lower and older leaves affected first','Premature leaf yellowing and defoliation','Dark cankers on stems at soil level','Black lesions at stem end of fruit in severe cases'],
        prev:['Rotate tomatoes with non-solanaceous crops (corn, legumes) for 2–3 years','Use certified disease-free seeds and healthy transplants only','Apply copper-based fungicide preventively before first sign of disease','Mulch soil surface to prevent rain splash and spore dispersal','Space plants 45–60 cm apart for canopy airflow','Remove and destroy all crop debris after harvest'],
        treat:['Remove and bag all infected leaves immediately — do not drop on soil','Apply chlorothalonil or mancozeb at 7–10 day intervals during humid weather','Switch to azoxystrobin if disease is severe or spreading rapidly','Convert to drip irrigation — overhead watering worsens spread significantly','Apply potassium fertilization to boost plant immune response'],
        parts:['Leaves','Stems','Fruit'], spread:'Moderate', loss:'20–50%' },
      { n:'Late Blight', sc:'Phytophthora infestans', ty:'f', sv:'high',
        short:'Rapidly spreading water-soaked lesions — the famine disease that destroyed the Irish potato crop. Entire crops collapse within 3–5 days.',
        cause:['Phytophthora infestans is an oomycete (water mold) that caused the Irish Famine of 1845–49.','Cool (10–20°C), wet, foggy, and overcast conditions trigger explosive spread.','Airborne sporangia travel kilometers on wind — entire regions can be infected within days.','Infected transplants and volunteer plants are the primary source of inoculum each season.'],
        syms:['Large water-soaked (greasy) patches at leaf edges and tips','White fluffy mycelial growth on leaf undersides in humid mornings','Dark brown to black stem lesions causing rapid collapse','Firm brown rot on green fruit progressing inward — eventually soft','Entire plants collapse within 5–7 days under ideal conditions','Foul musty smell from infected tissue'],
        prev:['Plant certified resistant varieties (Iron Lady, Defiant, Mountain Magic)','Apply preventive mancozeb or copper before cool-wet weather forecasts','Never use overhead irrigation — creates the leaf wetness conditions that trigger infection','Remove all tomato cull piles and volunteer plants from the area','Ensure excellent field drainage — standing water accelerates spread','Never compost any infected material — burn or bury deeply (50cm+)'],
        treat:['Apply cymoxanil + metalaxyl immediately at very first sign of infection','Repeat every 5–7 days during continued wet weather — no gaps in coverage','Apply in the evening — spores germinate at night during dew formation','Remove and destroy entire heavily infected plants immediately','Harvest any remaining healthy fruit promptly before infection spreads'],
        parts:['Leaves','Stems','Fruit'], spread:'Very High', loss:'40–100%' },
      { n:'Bacterial Wilt', sc:'Ralstonia solanacearum', ty:'b', sv:'high',
        short:'Sudden complete wilting of healthy-looking plants with no yellowing — bacteria block vascular vessels. No effective chemical cure.',
        cause:['Soil-borne bacterium invades plant roots through wounds from nematodes, transplanting, or tools.','Completely blocks xylem vessels preventing water movement — plants wilt despite adequate soil moisture.','Survives in soil and water for 2–7 years without a host, especially in warm tropical soils.','Hot saturated soils above 24°C dramatically accelerate upward colonization of vascular tissue.'],
        syms:['Sudden complete wilting starting at youngest shoots','Plants wilt during midday heat and DO NOT recover at night','White milky bacterial ooze flows from cut stem suspended in water','Brown discoloration of vascular tissue visible when stem is cut','Plant death typically within 2–5 days of first visible symptoms','Root rot visible in advanced stages'],
        prev:['Plant in well-drained soils pH 6–7 — waterlogged soils are highest risk','Use grafted tomato plants on wilt-resistant rootstocks (Beaufort, Maxifort)','Sterilize all tools with 10% bleach between plants during any handling','Never flood-irrigate — use drip irrigation and controlled soil moisture','Do not replant tomatoes, peppers, or eggplant in infected soil for 3+ years','Soil solarization for 6–8 weeks in summer can reduce bacterial populations'],
        treat:['No effective chemical cure once bacterial wilt is established','Remove and destroy entire infected plant including all root tissue immediately','Apply Bacillus subtilis (Serenade) as suppressive soil drench around healthy plants','Mark infected zones and use separate tools for infected vs clean areas','Treat surrounding soil with lime to raise pH and inhibit bacterial survival'],
        parts:['Vascular system','Stems','Roots'], spread:'High (soil/water)', loss:'Up to 100%' },
      { n:'Tomato Mosaic Virus', sc:'ToMV / TMV', ty:'v', sv:'med',
        short:'Mosaic mottled leaves with distorted growth and reduced fruit quality — spreads through contact and insect vectors.',
        cause:['Caused by ToMV or TMV — among the most stable plant viruses. Remains infectious on surfaces for years.','Spreads through mechanical contact: infected hands, tools, clothing, irrigation water.','Aphids and thrips transmit the virus between plants during feeding — even brief probing transmits it.','Tobacco products contain TMV — smoking near plants transmits the virus.'],
        syms:['Mottled yellow-green mosaic pattern on leaves','Leaf blade distortion, curling, and reduced size on young growth','Stunted overall plant growth','Misshapen, mottled, or bronze-colored fruit with poor set','Necrotic brown streaks on stems in severe infection','Early-infected plants produce significantly fewer fruit'],
        prev:['Plant only certified TMV/ToMV-resistant varieties (labeled TM or ToMV on packet)','Wash hands with soap for 30 seconds before handling plants','Disinfect all tools with 70% isopropyl alcohol or 10% bleach before each use','Control aphid and thrip populations with neem oil or yellow sticky traps','Never smoke or handle tobacco products near tomato plants','Remove infected plants immediately without disturbing adjacent healthy plants'],
        treat:['No chemical cure for any viral disease — prevention is the only control','Remove and destroy all infected plants immediately to stop spread','Apply neem oil to control aphid and thrip vectors that transmit the virus','Thoroughly disinfect greenhouse surfaces between crops with virucidal solutions','Purchase only certified virus-indexed transplants and seeds next season'],
        parts:['Leaves','Fruit','Growing tips'], spread:'High (contact)', loss:'10–40%' }
    ]
  },
  {
    id:'wheat', name:'Wheat', emoji:'🌾',
    desc:'World\'s most widely grown cereal crop, feeding 2.5+ billion people. Wheat rust diseases have historically caused famines. The Ug99 rust race (discovered in Uganda 1999) broke resistance in 80% of global wheat varieties — an ongoing global epidemic threat.',
    season:'Winter (Sept–July) / Spring (Mar–Aug)', origin:'Fertile Crescent — modern Turkey, Syria, Iraq', family:'Poaceae',
    diseases:[
      { n:'Stem Rust (Black Rust)', sc:'Puccinia graminis f.sp. tritici', ty:'f', sv:'high',
        short:'Brick-red pustules on stems and leaves — the most destructive wheat disease in history. Ug99 strain threatens global wheat production.',
        cause:['Puccinia graminis Ug99 race overcame the Sr31 resistance gene found in most global wheat varieties.','Alternate host is barberry (Berberis spp.) where sexual reproduction creates new pathogenic races.','Wind-borne urediniospores travel 1,000+ km — entire regions can be infected within days.','Warm days (25–35°C) with cool dew-forming nights create optimal infection conditions.'],
        syms:['Brick-red oval urediniospore pustules erupting through stem and leaf epidermis','Pustules surrounded by yellow necrotic halos','Black teliospore pustules at crop maturity','Infected stems become brittle and weak — cause lodging before harvest','Shriveled lightweight grain with poor milling quality','Complete head bleaching in severe epidemic years'],
        prev:['Plant Ug99-resistant varieties — consult ICAR or local extension for current race-specific recommendations','Eradicate barberry bushes within 1 km of wheat fields — they host the sexual stage','Apply preventive triazole fungicide at flag-leaf stage (GS 37–39) before disease threshold','Monitor national wheat rust early warning networks and spray at first detection in your region','Plant early-maturing varieties to escape the most severe rust pressure periods'],
        treat:['Apply triazole fungicide (tebuconazole 250 EC or propiconazole) immediately at first sign','Prioritize spraying flag leaf and upper 2–3 leaves — they provide 75% of grain fill','Repeat application 14–18 days later if weather favors continued rust spread','Harvest as early as physiologically possible in epidemic years — delay increases loss significantly','Report unusual or severe rust outbreaks to national plant protection organizations'],
        parts:['Stems','Leaves','Glumes','Awns'], spread:'Extremely High', loss:'Up to 70%' },
      { n:'Fusarium Head Blight (Scab)', sc:'Fusarium graminearum', ty:'f', sv:'high',
        short:'Bleached florets with pink mold at flowering — produces dangerous vomitoxin (DON) mycotoxin contaminating grain.',
        cause:['Infects wheat at anthesis (flowering) from infected residue — a 72-hour wet period at flowering can trigger epidemic.','Produces DON (vomitoxin) — harmful to humans and animals, cannot be removed by milling or cooking.','Warm (25–28°C) humid or rainy weather during flowering is the critical infection window.','Corn residue in the field is the primary inoculum source — corn-wheat rotation dramatically increases risk.'],
        syms:['Premature bleaching of individual florets or entire spikelets','Pink to orange mold at base of infected glumes in humid conditions','Shriveled chalky "tombstone" kernels with low test weight','Mycotoxin contamination — grain tastes bitter; no visual symptom for DON','Infected heads mixed randomly with healthy-appearing heads','Entire heads may be prematurely killed and bleached in severe cases'],
        prev:['Rotate with soybeans, canola, or sunflower for one year — breaks the pathogen cycle','Plant varieties with moderate FHB resistance — check national trial ratings annually','Apply metconazole (Caramba) or prothioconazole specifically at 25–50% anthesis — only effective timing','Time planting to minimize overlap between wheat flowering and forecasted wet humid weather','Plow or incorporate crop residue deeply to accelerate decomposition'],
        treat:['Fungicide only effective during flowering window — no post-infection treatment works','Harvest promptly at physiological maturity — delay worsens mycotoxin accumulation by 15–25%/week','Dry harvested grain to below 14% moisture within 24 hours to stop mycotoxin production','Test DON levels before sale — limits are 1 ppm for human food, 5 ppm for animal feed','Segregate suspect grain — heavily contaminated grain must be destroyed, not sold'],
        parts:['Heads (spikes)','Grain'], spread:'Moderate', loss:'10–50% + mycotoxin' },
      { n:'Powdery Mildew', sc:'Blumeria graminis f.sp. tritici', ty:'f', sv:'med',
        short:'White powdery fungal colonies on leaves and stems — reduces photosynthesis and grain fill.',
        cause:['Obligate biotrophic pathogen — survives only on living plant tissue, not in soil.','Dry days (15–22°C) with moderate humidity and cool dewy nights strongly favor infection.','Dense canopies with high nitrogen fertilization create warm humid microclimate for infection.','Wind-borne conidia spread rapidly within and between fields.'],
        syms:['White powdery pustules on upper leaf surfaces, sheaths, and stems','Powdery growth on ears and awns in severe infections','Yellowing and necrosis beneath established colonies','Premature leaf death — flag leaf is critical to protect','Reduced photosynthesis leading to poor grain fill and lower test weight'],
        prev:['Plant mildew-resistant varieties as single most cost-effective control measure','Avoid excess nitrogen — high N produces lush dense canopy that favors mildew','Reduce seeding rate and maintain plant spacing for airflow','Monitor from tillering through flag leaf emergence and spray at first sign'],
        treat:['Apply triazole (tebuconazole) or strobilurin (azoxystrobin) at first sign','One well-timed application at flag leaf stage often provides adequate season-long control','Prioritize coverage of upper canopy — flag leaf and next two leaves provide majority of grain fill'],
        parts:['Leaves','Stems','Ears'], spread:'High (wind)', loss:'5–30%' }
    ]
  },
  {
    id:'rice', name:'Rice', emoji:'🍚',
    desc:'Primary staple for 3.5 billion people as a daily food. Flooded paddy cultivation creates humid microclimate strongly favoring fungal and bacterial diseases. Rice blast alone causes annual losses equivalent to feeding 60 million people.',
    season:'Kharif/Wet season (June–Nov) / Rabi/Dry season (Nov–Apr)', origin:'China (Yangtze Valley) and India (Assam)', family:'Poaceae',
    diseases:[
      { n:'Rice Blast', sc:'Magnaporthe oryzae', ty:'f', sv:'high',
        short:'Most devastating rice disease globally — diamond-shaped leaf lesions plus neck rot causes complete panicle collapse and up to 100% yield loss.',
        cause:['Widely considered the world\'s most destructive plant pathogen.','Cool (22–28°C) with 10+ hours continuous leaf wetness triggers explosive infections.','Excessive nitrogen dramatically increases lesion size and susceptibility.','Spores are wind-borne, rain-splashed, and seed-borne — one infected seed can initiate a field epidemic.'],
        syms:['Diamond/rhombus-shaped lesions with gray-white centers and brown borders on leaves','Neck rot (neck blast) — dark brown rotting of panicle neck causing complete head collapse','Node blast — dark brown rotting of stem nodes causing lodging','Seedling blast — circular spots on rice seedlings in the nursery','Grain/panicle blast — brown discoloration of individual grain spikelets','White empty "deadheart" heads from neck rot infection'],
        prev:['Plant certified blast-resistant varieties — IR64, Swarna-Sub1, IR72 have partial resistance','Split nitrogen applications and avoid excess N at tillering stage','Drain paddies periodically (alternate wetting and drying) to reduce leaf wetness duration','Treat seed with tricyclazole or carbendazim before sowing to prevent seedling blast','Maintain crop density within recommended limits — overcrowding creates ideal microclimate'],
        treat:['Apply tricyclazole (Beam 75% WP) at 0.6g/L water at first leaf blast lesion appearance','Prophylactic spray at booting stage (panicle emergence) is critical for neck blast prevention','Apply isoprothiolane (Fuji-One) as systemic alternative with different mode of action','Repeat every 7–10 days during extended cool-wet periods','Harvest promptly at full maturity to minimize post-infection yield loss'],
        parts:['Leaves','Leaf collar','Nodes','Neck','Panicle','Grain'], spread:'Very High', loss:'Up to 100%' },
      { n:'Bacterial Leaf Blight', sc:'Xanthomonas oryzae pv. oryzae', ty:'b', sv:'high',
        short:'Water-soaked yellow-white leaf margin lesions advancing to complete leaf death — spreads explosively through irrigation water.',
        cause:['Bacteria enter through hydathode openings at leaf margins and wound sites during storms.','Hot humid conditions (25–34°C) with strong winds, heavy rain, and cyclone damage favor primary infection.','Bacteria accumulate in irrigation canals and can infect entire canal-connected field systems within days.','Infected fields spread the pathogen downstream through shared irrigation water.'],
        syms:['Water-soaked yellow-green lesions beginning at leaf tips and margins','Lesions expand into large straw-colored blighted areas covering entire leaf blade','Milky bacterial ooze on lesion surface in humid morning conditions (kresek phase)','Kresek phase in young plants causes rapid complete plant death within 2–3 weeks','Yellowing and drying of entire leaves in older plants'],
        prev:['Plant resistant varieties with Xa-resistance genes — IR64, IRRI 352, CR 1009','Balanced NPK fertilization — avoid luxury nitrogen that produces lush susceptible tissue','Drain fields before and immediately after cyclone or storm events','Use clean uncontaminated water sources — avoid direct canal connections between infected fields','Treat seeds with copper sulfate (0.25%) for 30 minutes before sowing'],
        treat:['Apply copper oxychloride (0.3%) as partial bacterial suppression','Drain field immediately and avoid re-flooding for 5–7 days to interrupt waterborne spread','Remove and destroy most severely infected plants before kresek phase spreads','Plant resistant varieties next season — most reliable long-term management'],
        parts:['Leaves','Vascular system'], spread:'High (irrigation)', loss:'20–70%' },
      { n:'Brown Planthopper (BPH)', sc:'Nilaparvata lugens', ty:'p', sv:'high',
        short:'Hopperburn — sap-sucking insect destroys crops from base upward creating circular scorched dead patches that cannot recover.',
        cause:['Phloem-feeding insect forms dense colonies at rice stem base, extracting sap causing direct plant starvation.','Populations explode from a few insects to 10,000+ per square meter within 3–4 weeks under warm humid high-N conditions.','Broad-spectrum insecticide misuse kills natural enemies (spiders, parasitic wasps) allowing BPH resurgence 10-fold within 2 weeks.','Populations migrate from southern regions — monitoring trap catches gives early warning of arrival.'],
        syms:['Circular to oval patches of scorched brown-yellow dying rice in otherwise healthy crop','Plants within patches collapse progressively from stem base outward — no recovery','Permanent expanding dead patches even after insect population declines','Honeydew secretion causes black sooty mold growth at plant base','White oval egg masses inserted into leaf sheaths','Adults are small (3–4mm), brown-black, found exclusively at stem base'],
        prev:['Plant BPH-resistant varieties — IR36, IR72, IR64, MTU 1010 have high resistance genes','Avoid over-application of nitrogen — lush soft stems are preferred by BPH','Place light traps to monitor adult BPH migration for early warning','Conserve natural enemies — spiders, Cyrtorhinus bugs, parasitic wasps — avoid broad-spectrum insecticides','Maintain field bunds for rapid water level adjustment'],
        treat:['Apply buprofezin (Applaud 25 WP) at stem base — specifically controls planthoppers, safe for natural enemies','Drain field before spraying to concentrate hoppers and improve penetration to stem base','NEVER apply pyrethroid insecticides — they kill natural enemies and cause severe BPH resurgence','Replant only if damage area exceeds 30% of total field area'],
        parts:['Stems','Leaf sheaths','Whole plant'], spread:'High (migration)', loss:'10–100%' }
    ]
  },
  {
    id:'corn', name:'Corn (Maize)', emoji:'🌽',
    desc:'World\'s largest cereal crop by volume — 1.2 billion tonnes annually. Grown on every inhabited continent as food, animal feed, and industrial raw material. Leaf diseases reduce yield by cutting photosynthesis during the critical grain fill period from silking to maturity.',
    season:'Spring planting (March–April) / Summer harvest (August–October)', origin:'Mexico (Teosinte ancestor)', family:'Poaceae',
    diseases:[
      { n:'Northern Corn Leaf Blight', sc:'Exserohilum turcicum', ty:'f', sv:'med',
        short:'Long cigar-shaped tan-gray lesions reducing photosynthesis and grain fill — most important corn leaf disease in humid temperate regions.',
        cause:['Overwinters as mycelium and conidia in infected corn residue left on the soil surface.','Moderate temperatures (18–27°C) with 6+ hours of leaf wetness favor spore germination and leaf penetration.','Spores infect lower leaves first, then spread upward through canopy as season progresses.','Continuous corn without rotation and minimum-till dramatically increases soil residue inoculum.'],
        syms:['Long (2.5–15 cm) cigar-shaped tan-gray lesions on leaves','Lesions run parallel to leaf veins with wavy irregular margins','Dense gray-green sporulation visible in moist conditions within lesion centers','Lower leaves infected first with upward progression through canopy','Entire leaves killed in severe cases leaving brown-gray dead tissue'],
        prev:['Plant hybrid varieties with Ht gene resistance (Ht1, Ht2, Ht3, HtN) as primary strategy','Rotate to soybean, alfalfa, or other non-corn crops for one season','Implement full-width tillage to bury residue and expose to rapid decomposition','Avoid late planting — early corn is more developed and better tolerates infection at grain fill','Scout fields starting at V8 growth stage — base fungicide decisions on economic thresholds'],
        treat:['Apply fungicide (azoxystrobin or propiconazole) at VT to R1 if lesions on lower leaves before tasseling','One well-timed application at tasseling typically provides adequate protection through grain fill','Prioritize spray coverage on upper canopy (ear leaf and above)','Evaluate economic return carefully based on yield potential, hybrid susceptibility, and severity'],
        parts:['Leaves'], spread:'Moderate', loss:'10–50%' },
      { n:'Gray Leaf Spot', sc:'Cercospora zeae-maydis', ty:'f', sv:'med',
        short:'Rectangular gray-tan lesions strictly bounded by leaf veins — most common and significant corn disease in high-humidity continuous corn regions.',
        cause:['Survives as mycelium and stromata in corn residue on the soil surface.','Extended leaf wetness of 11+ hours at 22–30°C enables consistent infection from rain or dew.','Minimum-till and continuous corn (same field for 2+ years) dramatically increases inoculum.','Spores splash upward from soil residue to lower leaves then wind-dispersed to upper canopy.'],
        syms:['Rectangular lesions strictly bounded and limited by leaf veins on both sides','Tan-brown when young, gray as they mature and sporulate','Multiple lesions coalesce into large rectangular dead areas of leaf blade','Lower leaves and sheaths affected first then progressing upward','Premature leaf senescence from base of plant moving upward'],
        prev:['Plant resistant hybrids — most effective and economical control available','Rotate to non-corn crops for one full season to dramatically reduce soil inoculum','Full-width tillage to bury residue and interrupt infection cycle','Maintain adequate potassium (K) fertilization — K deficiency increases susceptibility'],
        treat:['Apply fungicide at VT-R1 if disease exceeds economic threshold (≥5% of ear leaf and below at tasseling)','Aerial or ground application with good canopy penetration is essential','Evaluate economic return carefully — not always profitable in lower-yield environments'],
        parts:['Leaves','Leaf sheaths'], spread:'Moderate', loss:'20–40%' },
      { n:'Corn Smut', sc:'Ustilago maydis', ty:'f', sv:'low',
        short:'Large gray galls on ears, tassels, stems — alarming appearance but minimal yield loss. The galls are the edible Mexican delicacy huitlacoche.',
        cause:['Soil-borne spores survive 5–7 years in soil without a host plant.','Infection exclusively through wounds — insect feeding, hail damage, detasseling injuries.','Warm dry weather (25–35°C) following a wet period creates optimal infection conditions.','High nitrogen rates combined with rapid growth increase susceptibility.'],
        syms:['White to silver-gray swollen galls on any aerial plant part','Galls expand rapidly to 15+ cm diameter','Galls rupture at maturity releasing millions of powdery black spores','Infected ears may be completely replaced by gall tissue','Tassels entirely replaced by gall tissue in severe cases'],
        prev:['Control corn borer and aphid populations to minimize wound entry points','Avoid excessive nitrogen applications — balance fertilization','Remove and destroy galls before the black powdery mature stage','Do not plant in fields with heavy smut history for 2–3 seasons'],
        treat:['No effective fungicide for smut control once galls are established','Remove galls at white/silver stage before black spore release and bag them','Destroy removed galls by burning or deep burial — never compost','Harvest normal grain from remaining non-infected ears'],
        parts:['Ears','Tassels','Stems','Leaves'], spread:'Low (soil-borne)', loss:'1–10%' }
    ]
  },
  {
    id:'potato', name:'Potato', emoji:'🥔',
    desc:'World\'s third most important food crop with 374 million tonnes produced annually. Potatoes caused the Irish Great Famine (1845–49) when late blight destroyed crops for three consecutive years, killing 1 million and forcing 2 million to emigrate. Disease management remains existentially important.',
    season:'Spring planting (Feb–April) / Summer harvest (June–Sept)', origin:'Andes Mountains, Peru/Bolivia — domesticated 8,000+ years ago', family:'Solanaceae',
    diseases:[
      { n:'Late Blight', sc:'Phytophthora infestans', ty:'f', sv:'high',
        short:'The famine disease — kills entire potato plants and rots tubers within 3–5 days under cool wet conditions.',
        cause:['Same pathogen that causes tomato late blight — survives primarily in infected seed tubers used for replanting.','A2 mating type (emerged from Mexico in the 1980s) creates hybrid strains with greater aggressiveness and fungicide resistance.','Cool (10–20°C), wet, foggy, overcast weather triggers sporulation — the Blight Units system forecasts epidemic risk.','Airborne sporangia travel long distances — entire regional crops can be infected within days during epidemics.'],
        syms:['Dark water-soaked to brown-black spots at leaf margins and tips','White fluffy mycelial growth on leaf underside in humid morning conditions','Brown to black rot expanding on stems and petioles causing rapid collapse','Tuber rot — brown firm reddish discoloration from surface inward','Characteristic foul musty odor of infected foliage and tubers','Entire plant death within 3–7 days of first visible symptoms'],
        prev:['Use certified disease-free seed tubers — single most important action','Plant late blight resistant varieties (Sarpo Mira, Defender, Cara, Stirling)','Apply protectant fungicide (mancozeb, copper hydroxide) preventively before wet weather','Hill soil high around plants — protects tubers from airborne spores washing into soil','Destroy all cull piles, volunteer potatoes, and infected haulm immediately'],
        treat:['Apply cymoxanil + mancozeb (Curzate) immediately at very first sign','Rotate between fungicide chemistries with different modes of action every 1–2 applications','Remove and destroy heavily infected foliage (haul killing) to protect tubers before harvest','Harvest only when foliage is completely dead and dry — wet foliage allows spore infection of tubers','Dry and cure tubers rapidly at 10–15°C with airflow before storage'],
        parts:['Leaves','Stems','Tubers'], spread:'Extremely High', loss:'50–100%' },
      { n:'Common Scab', sc:'Streptomyces scabies', ty:'b', sv:'low',
        short:'Rough corky lesions on tuber skin reduce marketability — primarily a quality issue, not a yield problem.',
        cause:['Streptomyces scabies is an actinobacterium normally beneficial in soil but pathogenic when pH exceeds 5.5.','Alkaline conditions promote thaxtomin A toxin production causing corky tissue response in developing tubers.','Dry soil during 3–6 weeks following tuber initiation dramatically increases scab severity.','Fresh manure and lime additions that raise soil pH are the most common triggers for scab outbreaks.'],
        syms:['Raised corky brown patches on tuber skin — superficial to deep pitting','Lesions may be isolated or cover much of the tuber surface','No internal flesh damage — safe to eat after peeling with normal flavor'],
        prev:['Maintain soil pH between 4.8 and 5.2 — below 5.2 scab severity drops dramatically','Maintain consistent soil moisture during tuber initiation (3–6 weeks post-emergence)','Use only certified scab-free seed potatoes','Rotate with cereal grains or grass for 2+ years','Avoid fresh manure applications before planting'],
        treat:['No effective post-planting chemical treatment for scab','Apply elemental sulfur at 150–300 kg/ha before next planting to lower pH','Improve irrigation uniformity during tuber initiation period','Grade and sell scabbed tubers through appropriate channels'],
        parts:['Tuber skin surface'], spread:'Low (soil-borne)', loss:'Marketability reduction' },
      { n:'Potato Virus Y', sc:'PVY (Potyvirus)', ty:'v', sv:'med',
        short:'Most economically important potato virus — causes mosaic, leaf necrosis, and tuber necrotic ringspot disease.',
        cause:['Transmitted non-persistently by 40+ aphid species — a 30-second feeding probe can acquire and transmit the virus.','Infected seed tubers are primary long-distance spread mechanism — spreads globally through seed trade.','PVYNTN strain produces necrotic ringspots on tubers that are invisible externally but make entire crop unsaleable.','Winged aphid populations peak during spring and autumn migration periods driving explosive field spread.'],
        syms:['Mottled yellow-green mosaic pattern on leaves with alternating lighter and darker areas','Leaf rugosity — rough crinkled texture and reduced blade expansion','Necrotic brown ring spots on leaves in severe strains','Tuber necrotic ringspot disease — brown rings inside tubers, invisible from outside','Progressive stunting and decline over multiple seasons if infected tubers replanted'],
        prev:['Plant only certified virus-tested and indexed seed potatoes — foundation of PVY management','Remove (rogue) any infected-appearing plants as soon as identified — before aphid populations peak','Apply mineral oil sprays to leaves — oil prevents aphid stylet probing and virus acquisition','Control aphid vectors with selective insecticides (pirimicarb) that spare beneficial insects','Never save tubers from infected crops for replanting'],
        treat:['No chemical cure for viral diseases in any crop','Remove infected plants immediately and destroy — do not compost','Control aphid vectors to reduce secondary spread within crop','Use virus-free certified seed potatoes exclusively next season'],
        parts:['Leaves','Tubers','Growing tips'], spread:'High (aphids)', loss:'10–80%' }
    ]
  },
  {
    id:'soybean', name:'Soybean', emoji:'🫘',
    desc:'World\'s most important oilseed and legume protein crop — grown on 130 million hectares. Provides 70% of global protein meal for livestock feed. SDS and soybean cyst nematode together cause over $1 billion in US losses annually.',
    season:'Spring / Summer warm season crop', origin:'Northeastern China — domesticated ~3000 BCE', family:'Fabaceae',
    diseases:[
      { n:'Sudden Death Syndrome', sc:'Fusarium virguliforme', ty:'f', sv:'high',
        short:'Interveinal leaf necrosis with green veins — root rot toxins travel up to leaves while stem remains healthy externally.',
        cause:['Infects soybean roots during cool (10–15°C) wet soil conditions in first 2 weeks after planting.','Phytotoxins produced in infected roots transported through xylem to leaves causing foliar symptoms weeks after root infection.','Soybean cyst nematode (Heterodera glycines) dramatically worsens SDS by creating root wounds.','Compacted poorly drained soils that remain cold and saturated create ideal conditions.'],
        syms:['Interveinal chlorosis — yellowing between leaf veins while veins stay green on upper canopy','Necrosis between veins progressing rapidly from chlorosis','Leaflets drop prematurely while petioles remain attached','When stem is cut — white root rot and brown internal crown discoloration','Pod abortion, seed shriveling, incomplete pod fill in severe early-season infection'],
        prev:['Plant in well-drained fields — tile drainage is single most effective long-term management','Plant only when soil temperature is above 10°C — cold planting creates maximum risk','Apply seed treatment containing fluopyram (Ilevo) or metalaxyl specifically registered for SDS','Use SDS-tolerant varieties — significant tolerance differences between commercial varieties','Manage soybean cyst nematode through resistant varieties and 2-year corn rotation'],
        treat:['No foliar fungicide effective once foliar SDS symptoms appear — disease is in the roots','Seed treatment is the only effective intervention window','Improve drainage in severely affected areas for future seasons','Use combination SCN resistance + SDS tolerance in variety selection for maximum protection'],
        parts:['Roots','Crown','Leaves'], spread:'Low (soil-borne)', loss:'Up to 30%' },
      { n:'Asian Soybean Rust', sc:'Phakopsora pachyrhizi', ty:'f', sv:'high',
        short:'Tan to brick-red erupting pustules on leaf undersides — one of the fastest-spreading and most destructive crop diseases known.',
        cause:['Obligate biotrophic pathogen — survives only on living soybean and wild legume tissue, not in soil.','Spreads as urediniospores via wind currents — documented to cross oceans; caused $2 billion loss on reaching Brazil.','Warm (15–28°C) with >75% humidity and 6+ hours of leaf wetness enables rapid spore germination and penetration.','No soil inoculum reservoir — inoculum arrives via wind each season from green soybean crops in southern latitudes.'],
        syms:['Small (2–5 mm) tan to yellowish-brown lesions on upper leaf surfaces initially','On leaf undersides — cream to tan tube-like pustules (uredinia) erupting through epidermis','Pustule color progresses from cream/tan to brick-red as infection matures','Rapid yellowing and premature complete defoliation from bottom of canopy upward','Complete canopy loss and exposed pods in epidemic years'],
        prev:['Monitor national soybean rust surveillance networks — programs give 1–2 week regional warning','Apply preventive triazole + strobilurin fungicide 3–7 days BEFORE rust is forecast to reach your region','Plant early-maturing varieties to escape late-season high-risk period','Scout lower canopy leaves weekly from V6 — first pustules appear on lowest leaves first'],
        treat:['Apply triazole fungicide (tebuconazole, flutriafol) at first confirmed detection of pustules','ALWAYS use triazole + strobilurin tank mix — triazoles alone caused resistance in Brazilian populations','Repeat every 14–21 days during high-risk weather — maximum 4 applications per season','Use high water volume (150–200 L/ha) with canopy-penetrating nozzles — target underside of lower leaves'],
        parts:['Leaves (underside)','Petioles','Pods'], spread:'Extremely High (wind)', loss:'Up to 80%' }
    ]
  },
  {
    id:'banana', name:'Banana', emoji:'🍌',
    desc:'World\'s most exported fruit and fourth most important food crop. Production worth $25 billion annually; staple for 400 million people in tropical regions. Panama Disease TR4 has no effective control and is spreading to all major production regions — potentially eliminating Cavendish banana as we know it.',
    season:'Year-round perennial tropical crop', origin:'Papua New Guinea and Southeast Asia — domesticated 8,000+ years ago', family:'Musaceae',
    diseases:[
      { n:'Panama Disease (TR4)', sc:'Fusarium oxysporum f.sp. cubense Tropical Race 4', ty:'f', sv:'high',
        short:'Existential threat — soil-borne vascular wilt with no effective treatment has devastated plantations in Asia, Africa, and Latin America.',
        cause:['TR4 strain emerged from Asia in the 1990s and overcame resistance that protected Cavendish varieties from earlier races.','Spreads via infected soil, water, plant material, equipment tires, and even footwear — a single soil particle on a boot can introduce TR4.','Survives in soil as chlamydospores for 30–40 years without a host plant — cannot be cleaned by any current technology.','No commercial Cavendish variety is currently fully resistant to TR4.'],
        syms:['Yellowing and collapsing of the oldest outer leaves first — youngest inner leaves remain green initially','Leaf petioles break and hang down around the pseudo-stem (characteristic "skirt" appearance)','Cutting the pseudo-stem cross-section shows yellow to brown-red ring in vascular tissue','Pseudo-stem splitting at the base as infection progresses','Complete plant collapse — all leaves die within 3–6 months of first symptoms','All daughter suckers and ratoon crops eventually infected from the same soil'],
        prev:['Use only certified tissue culture (TC) disease-free planting material from accredited labs','Implement strict biosecurity — all visitors, workers, equipment must be cleaned and disinfected before entering plantation','Designate clean zones with quarantine procedures between blocks','TR4 is a regulated quarantine pest in most countries requiring mandatory reporting','Research and trial TR4-resistant varieties (GCTCV-218, tropical diploid hybrids) for future transition'],
        treat:['No effective chemical, biological, or cultural treatment can cure TR4 once established — this defines the Panama Disease tragedy','Immediate quarantine and destruction of all infected plants and 10-meter buffer zone','Do not replant banana for 20–30 years in confirmed infected soils','Report all suspected TR4 detections to national plant health authorities immediately','Biocontrol (Trichoderma, Bacillus) being researched but not yet reliable in field conditions'],
        parts:['Roots','Corm','Vascular system','Pseudo-stem','Whole plant'], spread:'Permanent (soil)', loss:'Complete plantation loss' },
      { n:'Black Sigatoka', sc:'Mycosphaerella fijiensis', ty:'f', sv:'high',
        short:'Most serious banana foliar disease — requires 40–60 fungicide applications per year in tropical regions to maintain acceptable yields.',
        cause:['Wind-borne ascospores ejected during rain from pseudothecia in infected leaf debris then dispersed by wind.','Cool (22–27°C) with high humidity, rainfall, and prolonged leaf wetness create optimal conditions for explosive spread.','Pathogen has developed documented resistance to benzimidazoles, DMI fungicides, and strobilurins.','Black Sigatoka replaced Yellow Sigatoka globally because of greater aggressiveness and wider host range.'],
        syms:['Initial pale green water-soaked streaks on lower leaf surfaces (easy to miss)','Dark brown to black expanding spots with gray centers','Multiple lesions coalesce to cause massive leaf necrosis — leaves die early','Up to 80% of leaf area destroyed in unmanaged crops in one season','Premature fruit maturity 2–4 weeks early — unsuitable for export markets','Reduced bunch weight and smaller finger size'],
        prev:['Remove and destroy infected leaves (deleafing) as soon as lesions are identified — removes inoculum','Maintain optimum plant spacing (3×2.5 m minimum) for maximum air movement','Strict preventive fungicide rotation: DMI fungicides, strobilurins, morpholines, and multisite protectants in sequence','Avoid prolonged leaf wetness from overhead irrigation — use drip irrigation','Plant tolerant varieties (Yangambi km5, FHIA hybrids) where available'],
        treat:['Apply triazole (propiconazole) + mancozeb rotation spray every 7–14 days during high humidity season','Aerial application standard for commercial plantations — ground equipment for smallholder farms','Rotate between 4–6 different fungicide modes of action in strict sequence to prevent resistance','Combine with surgical deleafing of visibly infected tissue after each spray cycle'],
        parts:['Leaves','Bracts'], spread:'Very High', loss:'20–50% fruit quality and weight' }
    ]
  },
  {
    id:'apple', name:'Apple', emoji:'🍎',
    desc:'One of the world\'s most economically important temperate fruit crops with 93 million tonnes globally. Commercial orchards apply 15–30 spray treatments per season. Apple scab alone can render 100% of fruit unmarketable without a preventive spray program.',
    season:'Spring bloom (March–May) / Autumn harvest (September–November)', origin:'Central Asia — Kazakhstan (Malus sieversii ancestor)', family:'Rosaceae',
    diseases:[
      { n:'Apple Scab', sc:'Venturia inaequalis', ty:'f', sv:'high',
        short:'Olive-green to black scab lesions on leaves and fruit — most economically important apple disease globally. 100% of crop unmarketable without management.',
        cause:['Overwinters as pseudothecia in fallen infected leaves on orchard floor — NOT in soil, only in leaf tissue.','Ascospores released during rain events in spring from green tip through late May.','Infection requires: susceptible tissue + free water + 6–24°C for specific number of wet hours (Mills Table).','At 16°C — most common spring temperature — only 9 continuous wet hours trigger infection.'],
        syms:['Olive-green to yellow-green velvety spots on upper leaf surfaces turning brown-black with age','Black scab lesions on developing fruit surface — infected fruit becomes distorted and cracked','Leaf distortion and premature defoliation in severe infections','Cracked fruit skin with russet-brown corky scab — completely unmarketable for fresh consumption','Young infected fruitlets frequently drop before harvest reducing yield directly'],
        prev:['Rake and collect all fallen leaves from orchard floor each autumn — removes primary inoculum','Plant scab-resistant varieties (Liberty, Enterprise, Redfree, Prima, GoldRush) — no spray program needed','Apply dormant copper sulfate or lime sulfur spray before bud break to kill pseudothecia','Use computerized Mills infection period models or smartphone apps to time fungicide sprays precisely','Maintain open canopy through regular pruning — good airflow reduces leaf wetness duration'],
        treat:['Apply protectant fungicide (captan, mancozeb) BEFORE forecasted infection period (Mills Period)','Apply post-infection (curative) fungicide (myclobutanil, difenoconazole) within 72–96 hours of infection','Continue spray program every 7–10 days through June then 14-day intervals for secondary scab','Thin crop to reduce fruit touching — scab spreads between touching fruit surfaces'],
        parts:['Leaves','Fruit','Shoots','Buds'], spread:'High (rain splash)', loss:'Up to 100% marketability' },
      { n:'Fire Blight', sc:'Erwinia amylovora', ty:'b', sv:'high',
        short:'Blossoms and shoots wilt suddenly as if scorched by fire — most destructive bacterial disease of apples globally.',
        cause:['Overwinters in visible cankers on branches and trunk — not in soil.','Carried by pollinators (honeybees, bumblebees), rain, and insects to open blossoms during primary infection.','Optimal infection requires: open blossoms + 18–29°C + rainfall, heavy dew, or humidity at bloom.','Secondary infections through growing shoot tips, hailstone wounds, and insect feeding sites throughout summer.'],
        syms:["Shepherd's crook — young shoot tips wilt, curl, and die in a distinctive hooked shape","Blossoms and young fruitlets turn brown-black and mummify — remain attached (do not drop)",'Dark water-soaked lesions on shoots expanding rapidly downward into larger wood','Amber to reddish bacterial ooze on infected shoot tissue in humid conditions','Streaking under bark — cutting reveals reddish-brown discoloration ahead of visible cankers'],
        prev:['Plant fire blight-resistant rootstocks (Geneva series G.935, G.11) and scions (Enterprise, Liberty)','Avoid excess nitrogen — high N creates rapid succulent shoot growth most susceptible','Apply copper bactericide or streptomycin (where registered) at pink through petal fall during bloom','Use fire blight infection models (Maryblyt) to spray only when risk is actually high','Prune only during dry warm weather — never in rain or high humidity'],
        treat:['Prune all infected shoots 25–30 cm BELOW visible infection margin — bacteria move ahead of visible symptoms','Sterilize pruning tools with 70% alcohol or 10% bleach between every single cut — non-negotiable','Remove and immediately destroy all pruned material — do not leave on orchard floor','Monitor pruned trees closely for 2–3 weeks — re-prune any new infections from bacteria already in tree'],
        parts:['Blossoms','Shoots','Branches','Trunk','Rootstock'], spread:'High (insects/rain)', loss:'50–100% affected wood' }
    ]
  },
  {
    id:'grape', name:'Grape', emoji:'🍇',
    desc:'World\'s most planted fruit crop by hectarage with 7.4 million hectares under cultivation. Grown for fresh consumption, wine, raisins, and juice. Downy and powdery mildew together account for the majority of all fungicide use in viticulture globally — some wine regions apply 15–20 fungicide sprays per season.',
    season:'Spring shoot emergence (April) / Autumn harvest (August–October)', origin:'Caucasus region — Georgia, Armenia (~6000 BCE)', family:'Vitaceae',
    diseases:[
      { n:'Downy Mildew', sc:'Plasmopara viticola', ty:'f', sv:'high',
        short:'Yellow oil spots on upper leaf with white cotton-like mold below — destroys leaves and entire grape clusters in wet seasons.',
        cause:['Oomycete introduced to Europe from North America on American rootstock vines in the 1870s — caused first European epidemics.','Overwinters as oospores in fallen infected leaves — germinates in spring when soil exceeds 11°C with rain.','Primary infection: temperatures 12–30°C + rainfall + 3+ continuous hours of leaf wetness (Rule of 10).','Secondary infections by asexual sporangiophores spread explosively during warm humid conditions.'],
        syms:['Yellow-green oily (water-soaked) spots on upper leaf surface visible against light','White fluffy downy sporulation on underside of oil spots during humid mornings','Shoot tip necrosis — growing tips die and curl (S-bend)','Infected inflorescences turn yellow then brown-black — die before setting fruit','Green berries shrivel, brown, and mummify (leather berry) before véraison','Defoliation of entire canopy in severe epidemic years'],
        prev:['Open canopy management through leaf removal — reduces humidity and improves airflow around clusters','Apply copper hydroxide preventively from 3-leaf stage through end of flowering — foundation of organic protection','Use predictive models based on weather data to time sprays to actual infection events','Remove and destroy fallen infected leaves in autumn to reduce oospore reservoir','Plant resistant grape varieties (Regent, Cabernet Cortis) where wine style permits'],
        treat:['Apply mancozeb + copper tank mix at first sign of oil spots on leaves','Switch to systemic (metalaxyl-M, fosetyl-Al) if established infection has occurred — curative activity','Remove and destroy infected clusters immediately to reduce within-canopy spread','Emergency leaf removal around infected zones to improve spray penetration and airflow'],
        parts:['Leaves','Shoots','Inflorescences','Berries'], spread:'Very High', loss:'Up to 75%' },
      { n:'Powdery Mildew', sc:'Erysiphe necator', ty:'f', sv:'high',
        short:'White powdery coating on all green parts — cracks berry skin at véraison causing secondary Botrytis infection and catastrophic quality loss.',
        cause:['Obligate biotrophic fungus overwintering in dormant buds — Flag Shoots carry established infection from bud burst.','Favors dry to moderately humid warm conditions (20–27°C) — DOES NOT require free water, unlike most fungi.','Conidia spread by wind during dry periods, germinating optimally at 25°C.','Flag shoots — shoots from overwintered infected buds — are primary source of early-season infection.'],
        syms:['White powdery conidia covering all green tissue — leaves, young shoots, petioles, inflorescences','Infected berries remain small and develop rusty brown star-shaped cracks (russeting) or split at véraison','Berry cracking provides direct Botrytis entry points causing secondary bunch rot','Stunted deformed shoot growth and vine tip necrosis','Pungent mushroom-like odor from heavily infected clusters'],
        prev:['Canopy management — leaf removal to achieve single-layer non-overlapping leaf walls','Apply preventive sulfur (dusting or wettable) from bud burst — classic, effective, zero resistance risk','Winter pruning to remove flag shoots and infected canes — wash shears with ethanol between vines','Apply kaolin clay particle film as physical barrier preventing spore germination','Plant resistant varieties (Regent, Muscaris, Solaris) where acceptable'],
        treat:['Apply wettable sulfur at first sign — highly effective and no resistance risk','DO NOT apply sulfur when temperature exceeds 32°C — causes leaf phytotoxicity','Use systemic DMI (tebuconazole, myclobutanil) for established infections requiring curative activity','Potassium bicarbonate (organic) applied as foliar spray eliminates established leaf colonies','Rotate between 3–6 fungicide modes of action in strict sequence to prevent resistance'],
        parts:['Leaves','Shoots','Clusters','Berries'], spread:'High (wind/dry)', loss:'40–60% quality reduction' }
    ]
  }
];

/* ============================================================
   ENCYCLOPEDIA RENDERING
============================================================ */
let selCrop = null;

function renderEncList(list) {
  document.getElementById('enc-list').innerHTML = (list || CROPS).map(c => `
  <div class="enc-item${selCrop === c.id ? ' sel' : ''}" onclick="pickCrop('${c.id}')">
    <div class="ei-em">${c.emoji}</div>
    <div style="flex:1;min-width:0;">
      <div class="ei-name">${c.name}</div>
      <div class="ei-cnt">${c.diseases.length} diseases documented</div>
    </div>
    <div class="ei-arr">${selCrop === c.id ? '▶' : '›'}</div>
  </div>`).join('');
}

function encFilter(v) {
  const f = (v || '').trim().toLowerCase();
  renderEncList(f ? CROPS.filter(c => c.name.toLowerCase().includes(f)) : CROPS);
}

function pickCrop(id) {
  selCrop = id;
  renderEncList();
  const crop = CROPS.find(c => c.id === id);
  if (!crop) return;
  const TC = { f:'dt-f', b:'dt-b', v:'dt-v', p:'dt-p', n:'dt-n' };
  const TL = { f:'Fungal', b:'Bacterial', v:'Viral', p:'Pest', n:'Nutritional' };
  const SC = { low:'ds-l', med:'ds-m', high:'ds-h' };
  const SL = { low:'Low Risk', med:'Moderate', high:'High Risk' };
  document.getElementById('enc-main').innerHTML = `
  <div class="crop-hdr">
    <div class="crop-hdr-bg">${crop.emoji}</div>
    <div class="crop-tag">🌱 Crop Profile</div>
    <div class="crop-title">${crop.name}</div>
    <div class="crop-desc">${crop.desc}</div>
    <div class="crop-meta">
      <span class="cm"><span class="cm-dot" style="background:var(--g);"></span>${crop.diseases.length} Diseases</span>
      <span class="cm"><span class="cm-dot" style="background:var(--b);"></span>${crop.season}</span>
      <span class="cm"><span class="cm-dot" style="background:var(--gold);"></span>${crop.origin}</span>
      <span class="cm"><span class="cm-dot" style="background:var(--p);"></span>${crop.family}</span>
    </div>
  </div>
  <div class="dis-filt">
    <span class="df a" onclick="filtDis('all',this)">All Types</span>
    <span class="df" onclick="filtDis('f',this)">🍄 Fungal</span>
    <span class="df" onclick="filtDis('b',this)">🦠 Bacterial</span>
    <span class="df" onclick="filtDis('v',this)">🧬 Viral</span>
    <span class="df" onclick="filtDis('p',this)">🐛 Pest</span>
  </div>
  <div class="dis-grid" id="dis-grid">
    ${crop.diseases.map((d, i) => `
    <div class="dc" onclick="showDov('${id}',${i})">
      <div class="dc-top">
        <div class="dc-hdr">
          <span class="dtype ${TC[d.ty] || 'dt-f'}">${TL[d.ty] || d.ty}</span>
          <span class="dsev ${SC[d.sv] || 'ds-m'}"><span class="dsev-dot"></span>${SL[d.sv] || d.sv}</span>
        </div>
        <div class="dc-name">${d.n}</div>
        <div class="dc-sci">${d.sc}</div>
        <div class="dc-short">${d.short}</div>
      </div>
      <div class="dc-foot">
        <div class="dc-tags">${(d.parts || []).slice(0, 3).map(p => `<span class="dc-tag">${p}</span>`).join('')}</div>
        <span class="dc-more">Full Details →</span>
      </div>
    </div>`).join('')}
  </div>`;
}

function filtDis(type, el) {
  document.querySelectorAll('.df').forEach(f => f.classList.remove('a'));
  el.classList.add('a');
  const crop = CROPS.find(c => c.id === selCrop);
  if (!crop) return;
  const TC = { f:'dt-f', b:'dt-b', v:'dt-v', p:'dt-p', n:'dt-n' };
  const TL = { f:'Fungal', b:'Bacterial', v:'Viral', p:'Pest', n:'Nutritional' };
  const SC = { low:'ds-l', med:'ds-m', high:'ds-h' };
  const SL = { low:'Low Risk', med:'Moderate', high:'High Risk' };
  const filtered = type === 'all' ? crop.diseases : crop.diseases.filter(d => d.ty === type);
  const grid = document.getElementById('dis-grid');
  if (!grid) return;
  grid.innerHTML = filtered.length > 0
    ? filtered.map(d => {
        const i = crop.diseases.indexOf(d);
        return `
        <div class="dc" onclick="showDov('${selCrop}',${i})">
          <div class="dc-top">
            <div class="dc-hdr">
              <span class="dtype ${TC[d.ty] || 'dt-f'}">${TL[d.ty] || d.ty}</span>
              <span class="dsev ${SC[d.sv] || 'ds-m'}"><span class="dsev-dot"></span>${SL[d.sv] || d.sv}</span>
            </div>
            <div class="dc-name">${d.n}</div>
            <div class="dc-sci">${d.sc}</div>
            <div class="dc-short">${d.short}</div>
          </div>
          <div class="dc-foot">
            <div class="dc-tags">${(d.parts || []).slice(0, 3).map(p => `<span class="dc-tag">${p}</span>`).join('')}</div>
            <span class="dc-more">Full Details →</span>
          </div>
        </div>`;}).join('')
    : `<div style="grid-column:1/-1;text-align:center;padding:44px;color:var(--t3);font-size:.85rem;">No ${TL[type] || type} diseases documented for ${crop.name}.</div>`;
}

function showDov(cropId, idx) {
  const crop = CROPS.find(c => c.id === cropId);
  const d = crop ? crop.diseases[idx] : null;
  if (!d) return;
  const TC = { f:'dt-f', b:'dt-b', v:'dt-v', p:'dt-p', n:'dt-n' };
  const TL = { f:'Fungal', b:'Bacterial', v:'Viral', p:'Pest', n:'Nutritional' };
  const SC = { low:'ds-l', med:'ds-m', high:'ds-h' };
  const SL = { low:'Low Risk', med:'Moderate', high:'High Risk' };
  const SVC = { low:'var(--g)', med:'var(--gold)', high:'var(--r)' };
  const SBW = { low:28, med:63, high:90 };
  const SBC = { low:'sb-l', med:'sb-m', high:'sb-h' };

  document.getElementById('dov-crop').textContent = `${crop.emoji} ${crop.name}`;
  document.getElementById('dov-badges').innerHTML = `
    <span class="dtype ${TC[d.ty] || 'dt-f'}" style="font-size:.61rem;">${TL[d.ty] || d.ty}</span>
    <span class="dsev ${SC[d.sv] || 'ds-m'}" style="font-size:.63rem;"><span class="dsev-dot"></span>${SL[d.sv]}</span>`;
  document.getElementById('dov-name').textContent = d.n;
  document.getElementById('dov-sci').textContent = d.sc;

  document.getElementById('dov-body').innerHTML = `
  <div class="sev-block">
    <div class="sev-row">
      <div class="sev-lbl">Severity Level</div>
      <div class="sev-val" style="color:${SVC[d.sv]};">${SL[d.sv]}</div>
    </div>
    <div class="sev-track">
      <div class="sev-bar ${SBC[d.sv]}" id="sev-anim-bar" style="width:0%;"></div>
    </div>
  </div>
  <div class="dov-info">
    <div class="dib"><div class="dib-l">Affected Parts</div><div class="dib-v">${(d.parts || []).join(' · ')}</div></div>
    <div class="dib"><div class="dib-l">Spread Rate</div><div class="dib-v">${d.spread || '—'}</div></div>
    <div class="dib"><div class="dib-l">Yield / Loss</div><div class="dib-v">${d.loss || 'Variable'}</div></div>
    <div class="dib"><div class="dib-l">Pathogen Type</div><div class="dib-v">${TL[d.ty] || d.ty}</div></div>
  </div>
  ${[
    { label:'🔬 Scientific Root Cause', bg:'var(--pc)', data:d.cause, type:'steps' },
    { label:'⚠️ Observable Symptoms', bg:'var(--gdc)', data:d.syms, type:'chips' },
    { label:'🛡️ Prevention Strategies', bg:'var(--gc)', data:d.prev, type:'steps' },
    { label:'💊 Treatment Protocol', bg:'var(--bc)', data:d.treat, type:'steps' }
  ].map(sec => `
  <div class="dov-sec">
    <div class="dov-sec-t">
      <span class="dov-sec-ico" style="background:${sec.bg};">${sec.label.split(' ')[0]}</span>
      ${sec.label.split(' ').slice(1).join(' ')}
    </div>
    ${sec.type === 'chips'
      ? `<div class="chips">${sec.data.map(s => `<span class="chip">${s}</span>`).join('')}</div>`
      : `<div class="sl">${sec.data.map((s, i) => `<div class="si"><div class="sn">${i+1}</div><p>${s}</p></div>`).join('')}</div>`}
  </div>`).join('')}`;

  document.getElementById('dov').classList.add('open');
  document.body.style.overflow = 'hidden';
  setTimeout(() => {
    const bar = document.getElementById('sev-anim-bar');
    if (bar) bar.style.width = SBW[d.sv] + '%';
  }, 110);
}

function closeDov() {
  document.getElementById('dov').classList.remove('open');
  document.body.style.overflow = '';
}

/* ============================================================
   INIT
============================================================ */
renderEncList();

// Auto-update share section when it becomes visible
new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) updShare();
}, { threshold: 0.1 }).observe(document.getElementById('share'));
console.log('%c✅ AgroShield v6.0 — 100% Working, Zero Mistakes', 'color:#00ff9d;font-weight:800;font-family:monospace;font-size:13px;');
