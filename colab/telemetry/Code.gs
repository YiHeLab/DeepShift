// DeepShift usage telemetry receiver: a Google Apps Script web app that appends one row per
// report to the active spreadsheet. Deploy: Extensions > Apps Script in a new Google Sheet,
// paste this file, Deploy > New deployment > Web app, "Execute as: Me", "Who has access:
// Anyone", copy the /exec URL into the notebook's TELEMETRY_URL (or DEEPSHIFT_TELEMETRY_URL).
// Only what the notebook prints under "telemetry ->" arrives here; a structure arrives only
// when the user set SHARE_STRUCTURE, and it is stored in its own sheet.
var COLUMNS = ["received_utc", "event", "timestamp_utc", "deepshift_version", "n_members",
               "device", "in_colab", "structure", "chain", "n_residues", "n_predictions",
               "seconds", "intervals"];

// The type of every field the notebook sends, and the longest string accepted. A record
// that does not match is refused and nothing is written.
var FIELDS = {event: ["string", 16], timestamp_utc: ["string", 40],
              deepshift_version: ["string", 40], n_members: ["number"],
              device: ["string", 8], in_colab: ["boolean"], structure: ["string", 16],
              chain: ["string", 4], n_residues: ["number"], n_predictions: ["number"],
              seconds: ["number"], intervals: ["boolean"]};

// A sheet cell holds at most 50,000 characters, so a shared structure is split across
// consecutive rows of CHUNK characters, each numbered "part" of "parts".
var CHUNK = 45000;
var MAX_STRUCTURE = 10000000;   // characters; a PDB-format file stays well below this

// A leading apostrophe stores a string as text, so a value beginning with "=", "+", "-"
// or "@" cannot become a formula.
function asText(value) {
  return typeof value === "string" ? "'" + value : value;
}

function problem(record) {
  if (record === null || typeof record !== "object" || Array.isArray(record)) {
    return "not a JSON object";
  }
  for (var key in FIELDS) {
    var type = FIELDS[key][0], value = record[key];
    if (typeof value !== type) return key + ": expected " + type;
    if (type === "number" && !isFinite(value)) return key + ": not a finite number";
    if (type === "string" && value.length > FIELDS[key][1]) return key + ": too long";
  }
  var pdb = record.structure_pdb;
  if (pdb !== undefined && (typeof pdb !== "string" || pdb.length > MAX_STRUCTURE)) {
    return "structure_pdb: expected a string of at most " + MAX_STRUCTURE + " characters";
  }
  return null;
}

function reply(body) {
  return ContentService.createTextOutput(JSON.stringify(body))
                       .setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var record;
  try {
    record = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply({ok: false, error: "not JSON"});
  }
  var refused = problem(record);
  if (refused) return reply({ok: false, error: refused});

  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var received = new Date().toISOString();
  var sheet = ss.getSheetByName("usage") || ss.insertSheet("usage");
  if (sheet.getLastRow() === 0) sheet.appendRow(COLUMNS);
  sheet.appendRow([received].concat(COLUMNS.slice(1).map(function (k) {
    return asText(record[k]);
  })));
  if (record.structure_pdb) {
    var shared = ss.getSheetByName("shared_structures") || ss.insertSheet("shared_structures");
    if (shared.getLastRow() === 0) {
      shared.appendRow(["received_utc", "structure", "chain", "part", "parts", "pdb_text"]);
    }
    var text = record.structure_pdb, parts = Math.ceil(text.length / CHUNK), rows = [];
    for (var i = 0; i < parts; i++) {
      rows.push([received, asText(record.structure), asText(record.chain), i + 1, parts,
                 asText(text.slice(i * CHUNK, (i + 1) * CHUNK))]);
    }
    shared.getRange(shared.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows);
  }
  return reply({ok: true});
}

function doGet() {
  return ContentService.createTextOutput("DeepShift telemetry receiver").setMimeType(ContentService.MimeType.TEXT);
}
