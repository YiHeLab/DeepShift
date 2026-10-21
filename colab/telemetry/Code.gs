// DeepShift usage telemetry receiver: a Google Apps Script web app that appends one row per
// report to the active spreadsheet. Deploy: Extensions > Apps Script in a new Google Sheet,
// paste this file, Deploy > New deployment > Web app, "Execute as: Me", "Who has access:
// Anyone", copy the /exec URL into the notebook's TELEMETRY_URL (or DEEPSHIFT_TELEMETRY_URL).
// Only what the notebook prints under "telemetry ->" arrives here; a structure arrives only
// when the user set SHARE_STRUCTURE, and it is stored in its own sheet.
var COLUMNS = ["received_utc", "event", "timestamp_utc", "deepshift_version", "n_members",
               "device", "in_colab", "structure", "chain", "n_residues", "n_predictions",
               "seconds", "intervals"];

function doPost(e) {
  var record = JSON.parse(e.postData.contents);
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName("usage") || ss.insertSheet("usage");
  if (sheet.getLastRow() === 0) sheet.appendRow(COLUMNS);
  var row = [new Date().toISOString()].concat(COLUMNS.slice(1).map(function (k) {
    return record[k] === undefined ? "" : record[k];
  }));
  sheet.appendRow(row);
  if (record.structure_pdb) {
    var shared = ss.getSheetByName("shared_structures") || ss.insertSheet("shared_structures");
    shared.appendRow([new Date().toISOString(), record.structure || "", record.chain || "",
                      record.structure_pdb]);
  }
  return ContentService.createTextOutput(JSON.stringify({ok: true}))
                       .setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  return ContentService.createTextOutput("DeepShift telemetry receiver").setMimeType(ContentService.MimeType.TEXT);
}
